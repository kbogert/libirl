module occlusion;

import discretemdp;
import discretefunctions;
import trajectories;
import solvers;
import std.typecons;
import std.algorithm.comparison;
import std.random;
import std.stdio;


public Set!State randomOccludedStates(Model m, int num_occluded_states) {

    Set!State returnval = new Set!(State)(new Tuple!(State)[0]);

    auto arr = m.S().toArray().randomShuffle();

    return new Set!(State)(arr[0..num_occluded_states]);
    
}

public Sequence!(State, Action) removeOccludedTimesteps (Sequence!(State, Action) trajectory, Set!State occluded_states) {

    Sequence!(State, Action) returnval = new Sequence!(State, Action)(trajectory.length);

    foreach(i, timestep ; trajectory) {
        if (! occluded_states.contains(timestep[0])) {
            returnval[i] = timestep;
        } else {
            returnval[i] = Tuple!(State, Action)(null, null);
        }
    }
    
    return returnval;
}

public Sequence!(State, Action)[] removeOccludedTimesteps (Sequence!(State, Action)[] trajectory, Set!State occluded_states) {

    Sequence!(State, Action)[] returnval = new Sequence!(State, Action)[trajectory.length];

    foreach (i ; 0 .. trajectory.length) {
        returnval[i] = removeOccludedTimesteps(trajectory[i], occluded_states);
    }

    return returnval;   
}



// compute the trajectory distribution assuming missing nodes are caused by occlusion.
// the missing states could only be those which are in the set of occluded states
class ExactOccludedTrajectoryToTrajectoryDistr : ExactPartialTrajectoryToTrajectoryDistr {

    protected Distribution!(State)[] occluded_states_distr;

    
    public this(Model m, LinearReward r, Set!State[] occluded_states_array, bool extend_terminals_to_equal_length = true) {

        occluded_states_distr = new Distribution!(State)[occluded_states_array.length];

        foreach (i, occluded_states ; occluded_states_array) {        
            occluded_states_distr[i] = new Distribution!(State)(m.S(), 0.0);
            foreach( s ; occluded_states) {
                occluded_states_distr[i][s[0]] = 1.0 / occluded_states.size();
            }
        }
        super(m, r, extend_terminals_to_equal_length);
    }

    override protected Distribution!(State, Action) first_timestep(Distribution!(State) initial, ConditionalDistribution!(Action, State) policy, size_t timestep) {

        auto returnval = policy * new Distribution!(State)(initial * occluded_states_distr[timestep]);
//        returnval.normalize();
        return returnval;

    }

    override protected Distribution!(State, Action) forward_timestep(Distribution!(State, Action) previous_timestep, ConditionalDistribution!(Action, State) policy, size_t timestep) {

        auto returnval = policy * new Distribution!(State)(sumout!(Action)(sumout!(State)( (m.T() * previous_timestep).reverse_params())) * occluded_states_distr[timestep]);
//        returnval.normalize();
        return returnval;

    }

    override protected Distribution!(State, Action) reverse_timestep(Distribution!(State, Action) current_timestep, Distribution!(State, Action) next_timestep, size_t timestep) {
        current_timestep = new Distribution!(State, Action)((current_timestep.reverse_params() * occluded_states_distr[timestep]).reverse_params());
        auto returnval = new Distribution!(State, Action)(sumout!(State)( ((m.T() * current_timestep) * sumout!(Action)(next_timestep ) ) ) );
//        returnval.normalize();
        return returnval;
    }

}

// for occlusion that doesn't change each timestep
class ExactStaticOccludedTrajectoryToTrajectoryDistr : ExactOccludedTrajectoryToTrajectoryDistr {

    public this(Model m, LinearReward r, Set!State occluded_states, size_t max_trajectory_length, bool extend_terminals_to_equal_length = true) {

        super(m, r, new Set!State[0], extend_terminals_to_equal_length);
        
        occluded_states_distr = new Distribution!(State)[max_trajectory_length];
        auto temp = new Distribution!(State)(m.S(), 0.0);
        foreach( s ; occluded_states) {
            try {
                temp[s[0]] = 1.0 / occluded_states.size();
            } catch (Exception e) {
                writeln(s[0], " ", m.S());
                throw e;
            }
        }
        occluded_states_distr[] = temp;

    }
}

// alternative to my implementation using a markov smoother
abstract class MCMCOccludedTrajectoryToTrajectoryDistr: Sequence_Distribution_Computer!(State, Action) {

    protected bool extend_terminals_to_equal_length;
    protected Model m;
    protected LinearReward r;
    protected Set!State[] occluded_states;
    protected size_t repeats;
    
    public this(Model m, LinearReward r, size_t repeats, Set!State[] occluded_states, bool extend_terminals_to_equal_length = true) {
        this.extend_terminals_to_equal_length = extend_terminals_to_equal_length;
        this.m = m;
        this.r = r;
        this.occluded_states = occluded_states;
        this.repeats = repeats;
    }

    public Sequence!(Distribution!(State, Action))[] to_traj_distr( Sequence!(State, Action)[] trajectories, double [] weights ) {

        r.setWeights(weights);
        m.setR(r.toFunction());
        
        auto policy = m.getPolicy();

        auto full_space = m.S().cartesian_product(m.A());        
        auto tuple_full_space = pack_set(full_space);

        Sequence!(Distribution!(State, Action))[] returnval = new Sequence!(Distribution!(State, Action))[trajectories.length];

        
        foreach (i, traj ; trajectories) {
            
            // build an observation sequence

             Sequence!(Distribution!(Tuple!(State, Action))) observations = new Sequence!(Distribution!(Tuple!(State, Action)))(traj.length);
             foreach( t, timestep; traj) {

                Distribution!(Tuple!(State, Action)) dist;

                if (timestep[0] is null) {
                    // state is missing
                    auto temporary = new Distribution!(State)(m.S(), 0.0);
                    foreach (s; occluded_states[t]) {
                        // only the occluded states could be in the missing timesteps
                        temporary[s] = 1.0 / occluded_states[t].size();
                    }
                    dist = pack_distribution(policy * temporary);
                   
                } else if (timestep[1] is null) {
                    // action is missing
                    auto temp_state_dist = new Distribution!(State)(m.S(), 0.0);
                    temp_state_dist[timestep[0]] = 1.0;
                    dist = pack_distribution(policy * temp_state_dist);
                    
                } else {
                    // neither are missing
                    dist = new Distribution!(Tuple!(State, Action))(tuple_full_space, 0.0);
                    dist[timestep] = 1.0;
                }
                

                observations[t] = tuple(dist);
             }

             
            // build a transition function
            
            auto transitions = new ConditionalDistribution!(Tuple!(State, Action), Tuple!(State, Action))(tuple_full_space, tuple_full_space);
            foreach (sa ; full_space) {
                transitions[sa] = pack_distribution( policy * m.T()[sa] );
                
            }

            Distribution!(Tuple!(State, Action)) initial_state = pack_distribution(policy * m.initialStateDistribution());

            Sequence!(Function!(double, State, Action)) sum_results = new Sequence!(Function!(double, State, Action))(observations.length);
            foreach(ref timestep; sum_results) {
                timestep = tuple(new Function!(double, State, Action)(full_space, 0.0));
            }
            
            foreach( repeat; 0 .. repeats) {
                
                auto results = call_solver(observations, transitions, initial_state, i);

                foreach(t, ref timestep; sum_results) {
                    timestep = tuple(timestep[0] + results[t][0]);
                }
                
            }

            Sequence!(Distribution!(State, Action)) avg_results = new Sequence!(Distribution!(State, Action))(observations.length);                    
            foreach(t, timestep; sum_results) {
                avg_results[t] = tuple(new Distribution!(State, Action)(timestep[0]));
                avg_results[t][0].normalize();
            }

            returnval[i] = avg_results;

        }

        return returnval;                
    }

    protected abstract Sequence!(Distribution!(State, Action)) call_solver(Sequence!(Distribution!(Tuple!(State, Action))) observations, ConditionalDistribution!(Tuple!(State, Action), Tuple!(State, Action)) transitions, Distribution!(Tuple!(State, Action)) initial_state, size_t traj_num);

}

class MarkovSmootherExactOccludedTrajectoryToTrajectoryDistr : MCMCOccludedTrajectoryToTrajectoryDistr {
    
    public this(Model m, LinearReward r, Set!State[] occluded_states, bool extend_terminals_to_equal_length = true) {
        super(m, r, 1, occluded_states, extend_terminals_to_equal_length);
    }

    protected override Sequence!(Distribution!(State, Action)) call_solver(Sequence!(Distribution!(Tuple!(State, Action))) observations, ConditionalDistribution!(Tuple!(State, Action), Tuple!(State, Action)) transitions, Distribution!(Tuple!(State, Action)) initial_state, size_t traj_num) {

        auto temp_sequence = SequenceMarkovChainSmoother!(Tuple!(State, Action))(observations, transitions, initial_state);
        auto results = new Sequence!(Distribution!(State, Action))(temp_sequence.length);

        foreach (t, timestep; temp_sequence) {
            results[t] = tuple(unpack_distribution(timestep[0]));
        }
        return results;

    }
    
}

class GibbsSamplingApproximateOccludedTrajectoryToTrajectoryDistr: MCMCOccludedTrajectoryToTrajectoryDistr {

    protected size_t burn_in_samples;
    protected size_t num_samples;
        
    public this(Model m, LinearReward r, size_t repeats, Set!State[] occluded_states, size_t burn_in_samples, size_t num_samples, bool extend_terminals_to_equal_length = true) {
        super(m, r, repeats, occluded_states, extend_terminals_to_equal_length);

        this.burn_in_samples = burn_in_samples;
        this.num_samples = num_samples;
    }

    protected override Sequence!(Distribution!(State, Action)) call_solver(Sequence!(Distribution!(Tuple!(State, Action))) observations, ConditionalDistribution!(Tuple!(State, Action), Tuple!(State, Action)) transitions, Distribution!(Tuple!(State, Action)) initial_state, size_t traj_num) {

        auto temp_sequence = MarkovGibbsSampler!(Tuple!(State, Action))(observations, transitions, initial_state, burn_in_samples, num_samples);
        auto results = new Sequence!(Distribution!(State, Action))(temp_sequence.length);

        foreach (t, timestep; temp_sequence) {
            results[t] = tuple(unpack_distribution(timestep[0]));
        }

        return results;        

    }
}


class HybridMCMCApproximateOccludedTrajectoryToTrajectoryDistr: MCMCOccludedTrajectoryToTrajectoryDistr {

    protected size_t burn_in_samples;
    protected size_t num_samples;
    protected Sequence!(Distribution!(Tuple!(State, Action)))[] proposalDistributions;
    protected bool use_adaptive;
            
    public this(Model m, LinearReward r, size_t repeats, Set!State[] occluded_states, size_t burn_in_samples, size_t num_samples, Sequence!(Distribution!(State, Action))[] proposalDistributions, bool useAdaptiveProposalDistribution = true, bool extend_terminals_to_equal_length = true) {
        super(m, r, repeats, occluded_states, extend_terminals_to_equal_length);

        this.burn_in_samples = burn_in_samples;
        this.num_samples = num_samples;
        this.use_adaptive = useAdaptiveProposalDistribution;

        this.proposalDistributions = new Sequence!(Distribution!(Tuple!(State, Action)))[proposalDistributions.length];
        foreach(i; 0 .. proposalDistributions.length) {
            this.proposalDistributions[i] = new Sequence!(Distribution!(Tuple!(State, Action)))(proposalDistributions[i].length);
            foreach(t, timestep; proposalDistributions[i]) {
                this.proposalDistributions[i][t] = tuple(pack_distribution(timestep[0]));
            }
        }
    }

    protected override Sequence!(Distribution!(State, Action)) call_solver(Sequence!(Distribution!(Tuple!(State, Action))) observations, ConditionalDistribution!(Tuple!(State, Action), Tuple!(State, Action)) transitions, Distribution!(Tuple!(State, Action)) initial_state, size_t traj_num) {

        Sequence!(Distribution!(Tuple!(State, Action))) temp_sequence;

        if (use_adaptive) {
            temp_sequence = AdaptiveHybridMCMC!(Tuple!(State, Action))(observations, transitions, initial_state, proposalDistributions[traj_num], burn_in_samples, num_samples);
        } else {
            temp_sequence = HybridMCMC!(Tuple!(State, Action))(observations, transitions, initial_state, proposalDistributions[traj_num], burn_in_samples, num_samples);
        }
        
        auto results = new Sequence!(Distribution!(State, Action))(temp_sequence.length);

        foreach (t, timestep; temp_sequence) {
            results[t] = tuple(unpack_distribution(timestep[0]));
        }

        return results;        

    }
}



// for occlusion that doesn't change each timestep
class MarkovSmootherExactStaticOccludedTrajectoryToTrajectoryDistr: MarkovSmootherExactOccludedTrajectoryToTrajectoryDistr {

    public this(Model m, LinearReward r, Set!State occluded_states, size_t max_trajectory_length, bool extend_terminals_to_equal_length = true) {

        Set!State [] oc = new Set!State[max_trajectory_length];
        oc[] = occluded_states;
        
        super(m, r, oc, extend_terminals_to_equal_length);
    }
}

class GibbsSamplingApproximateStaticOccludedTrajectoryToTrajectoryDistr: GibbsSamplingApproximateOccludedTrajectoryToTrajectoryDistr {

    public this(Model m, LinearReward r, size_t repeats, Set!State occluded_states, size_t max_trajectory_length, size_t burn_in_samples, size_t num_samples, bool extend_terminals_to_equal_length = true) {

        Set!State [] oc = new Set!State[max_trajectory_length];
        oc[] = occluded_states;

        super(m, r, repeats, oc, burn_in_samples, num_samples, extend_terminals_to_equal_length);
        
    }
}

class HybridMCMCApproximateStaticOccludedTrajectoryToTrajectoryDistr: HybridMCMCApproximateOccludedTrajectoryToTrajectoryDistr {

    public this(Model m, LinearReward r, size_t repeats, Set!State occluded_states, size_t max_trajectory_length, size_t burn_in_samples, size_t num_samples, Sequence!(Distribution!(State, Action))[] proposalDistributions, bool useAdaptiveProposalDistribution = true, bool extend_terminals_to_equal_length = true) {

        Set!State [] oc = new Set!State[max_trajectory_length];
        oc[] = occluded_states;

        super(m, r, repeats, oc, burn_in_samples, num_samples, proposalDistributions, useAdaptiveProposalDistribution , extend_terminals_to_equal_length);
        
    }
}
