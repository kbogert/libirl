module trajectories;

import discretefunctions;
import discretemdp;
import solvers;
import std.typecons;
import partialobservation;
import std.math;
import std.random;


// Takes in incomplete trajectories and computes the complete distribution over all possible trajectories
interface Sequence_Distribution_Computer(T ...) {

    public Sequence!(Distribution!(T))[] to_traj_distr( Sequence!(T)[] trajectories, double [] weights );
}


// Takes perfectly observed trajectories and converts them to trajectory distributions
class CompleteTrajectoryToTrajectoryDistr : Sequence_Distribution_Computer!(State, Action) {

    Set!(State) state_space;
    Set!(Action) action_space;
    bool extend_terminals_to_equal_length;
    
    public this(Set!(State) state_set, Set!(Action) action_set, bool extend_terminals_to_equal_length = true) {
        this.state_space = state_set;
        this.action_space = action_set;
        this.extend_terminals_to_equal_length = extend_terminals_to_equal_length;
    }

    public Sequence!(Distribution!(State, Action))[] to_traj_distr( Sequence!(State, Action)[] trajectories, double [] weights ) {

        Sequence!(Distribution!(State, Action))[] returnval = new Sequence!(Distribution!(State, Action))[trajectories.length];

        auto full_space = state_space.cartesian_product(action_space);
        size_t max_traj_length = 0;
        foreach (t ; trajectories)
            if (t.length() > max_traj_length)
                max_traj_length = t.length();
        
        foreach(i, traj ; trajectories) {

            Sequence!(Distribution!(State, Action)) seq = new Sequence!(Distribution!(State, Action))(traj.length());
            foreach(t, timestep ; traj) {
                Distribution!(State, Action) dist = new Distribution!(State, Action)(full_space, 0.0);

            
                if (timestep[0].isTerminal() && timestep[1] is null) {
                    // In this case, we're missing the action for the terminal state
                    // fix this by making all actions equally likely
                    foreach (a ; action_space) {
                        dist[tuple(timestep[0], a[0])] = 1.0 / action_space.size();
                    }
                    
                } else {
                    dist[timestep] = 1.0;
                }

                seq[t] = tuple(dist);
            }

            if (traj[$][0].isTerminal() && extend_terminals_to_equal_length ) {

                while( seq.length() < max_traj_length) {
                    seq ~= seq[$];
                }
            }
            
            returnval[i] = seq;
        }

        return returnval;

    }

}

// convenience, since these types of trajectories are generated by so many tests
public Sequence!(Distribution!(State, Action))[] traj_to_traj_distr( Sequence!(State, Action)[] trajectories, Model m, bool extend_to_equal_lengths = true) {

    CompleteTrajectoryToTrajectoryDistr converter = new CompleteTrajectoryToTrajectoryDistr(m.S(), m.A());

    return converter.to_traj_distr(trajectories, null);
    
}

// ******************************************************************************
// Trajectories with missing entries (random data loss)
// ******************************************************************************


// Takes partial trajectories and converts them to trajectory distributions
// Exact version, interates through all possible trajectories using a forward-backword algorithm
class ExactPartialTrajectoryToTrajectoryDistr : Sequence_Distribution_Computer!(State, Action) {

    protected bool extend_terminals_to_equal_length;
    protected Model m;
    protected LinearReward r;
    
    public this(Model m, LinearReward r, bool extend_terminals_to_equal_length = true) {
        this.extend_terminals_to_equal_length = extend_terminals_to_equal_length;
        this.m = m;
        this.r = r;
    }

    public Sequence!(Distribution!(State, Action))[] to_traj_distr( Sequence!(State, Action)[] trajectories, double [] weights ) {

        r.setWeights(weights);
        m.setR(r.toFunction());
        
        auto policy = m.getPolicy();

        auto full_space = m.S().cartesian_product(m.A());
        
        Sequence!(Distribution!(State, Action))[] forward = new Sequence!(Distribution!(State, Action))[trajectories.length];

        size_t max_traj_length = 0;
        foreach (t ; trajectories)
            if (t.length() > max_traj_length)
                max_traj_length = t.length();

        // forward step
        foreach(i, traj ; trajectories) {

            Sequence!(Distribution!(State, Action)) seq = new Sequence!(Distribution!(State, Action))(traj.length());
            foreach(t, timestep ; traj) {
                Distribution!(State, Action) dist;

                if (timestep[0] is null) {
                    // state is missing
                    if (t == 0) {
                        // first timestep, use initial state distribution
                        dist = first_timestep(m.initialStateDistribution(), policy, t);
                    } else {
                        dist = forward_timestep(seq[t-1][0], policy, t);
                    }
                    
                } else if (timestep[1] is null) {
                    // action is missing
                    auto temp_state_dist = new Distribution!(State)(m.S(), 0.0);
                    temp_state_dist[timestep[0]] = 1.0;
                    dist = policy * temp_state_dist;
                    
                } else {
                    // neither are missing
                    dist = new Distribution!(State, Action)(full_space, 0.0);
                    dist[timestep] = 1.0;
                }

                seq[t] = tuple(dist);
            }

            if (traj[$][0] ! is null && traj[$][0].isTerminal() && extend_terminals_to_equal_length ) {

                while( seq.length() < max_traj_length) {
                    seq ~= seq[$];
                }
            }
            
            forward[i] = seq;
        }
        
        Sequence!(Distribution!(State, Action))[] reverse = new Sequence!(Distribution!(State, Action))[trajectories.length];
        // reverse step
        foreach(i, traj ; trajectories) {
            Sequence!(Distribution!(State, Action)) seq = new Sequence!(Distribution!(State, Action))(traj.length());
            seq[$] = tuple(new Distribution!(State, Action)(full_space, 1.0));
            
            foreach_reverse(t, timestep ; traj) {

                if (trajectories[i].length > t) {

                    Distribution!(State, Action) dist;

                    if (timestep[0] is null) {
                        // state is missing
                        auto temp = policy * new Distribution!(State)(m.S(), DistInitType.Uniform);
                        if (t < traj.length - 1) {
                            dist = reverse_timestep(temp, seq[t+1][0], t);
                        } else {
                            dist = reverse_timestep(temp, new Distribution!(State, Action)(full_space, 1.0), t);
                        }        
                    } else if (timestep[1] is null) {
                        // action is missing
                        auto temp_state_dist = new Distribution!(State)(m.S(), 0.0);
                        temp_state_dist[timestep[0]] = 1.0;
                        dist = policy * temp_state_dist;
                    
                    } else {
                        // neither are missing
                        dist = new Distribution!(State, Action)(full_space, 0.0);
                        dist[timestep] = 1.0;
                    }

                    seq[t] = tuple(dist);
                }
                if (traj[$][0] ! is null && traj[$][0].isTerminal() && extend_terminals_to_equal_length ) {

                    while( seq.length() < max_traj_length) {
                        seq ~= seq[$];
                    }
                }
            }
            reverse[i] = seq;            
            
        }

        Sequence!(Distribution!(State, Action))[] returnval = new Sequence!(Distribution!(State, Action))[trajectories.length];
        // combine together
        foreach(i, traj ; forward) {
            returnval[i] = new Sequence!(Distribution!(State, Action))(traj.length());
            foreach( t, timestep; traj) {
                returnval[i][t] = tuple(new Distribution!(State, Action)(timestep[0] * reverse[i][t][0]));
                returnval[i][t][0].normalize();
            }
        }
        
        return returnval;

    }

    protected Distribution!(State, Action) first_timestep(Distribution!(State) initial, ConditionalDistribution!(Action, State) policy, size_t timestep) {

        auto returnval = policy * initial;
//        returnval.normalize();        
        return returnval;

    }
    
    protected Distribution!(State, Action) forward_timestep(Distribution!(State, Action) previous_timestep, ConditionalDistribution!(Action, State) policy, size_t timestep) {

        auto returnval = policy * new Distribution!(State)(sumout!(Action)(sumout!(State)( (m.T() * previous_timestep).reverse_params())));
//        returnval.normalize();        
        return returnval;

    }

    protected Distribution!(State, Action) reverse_timestep(Distribution!(State, Action) current_timestep, Distribution!(State, Action) next_timestep, size_t timestep) {
        auto returnval = new Distribution!(State, Action)(sumout!(State)( ((m.T() * current_timestep) * sumout!(Action)(next_timestep ) ) ) );
//        returnval.normalize();
        return returnval;
    }
}


// base class for alternatives to my implementations using MCMC solvers  
abstract class MCMCPartialTrajectoryToTrajectoryDistr: Sequence_Distribution_Computer!(State, Action) {

    protected bool extend_terminals_to_equal_length;
    protected Model m;
    protected LinearReward r;
    protected size_t repeats;
        
    public this(Model m, LinearReward r, size_t repeats, bool extend_terminals_to_equal_length = true) {
        this.extend_terminals_to_equal_length = extend_terminals_to_equal_length;
        this.m = m;
        this.r = r;
        this.repeats = repeats;
    }

    public Sequence!(Distribution!(State, Action))[] to_traj_distr( Sequence!(State, Action)[] trajectories, double [] weights ) {

        r.setWeights(weights);
        m.setR(r.toFunction());
        
        auto policy = m.getPolicy();

        auto full_space = m.S().cartesian_product(m.A());        
        auto tuple_full_space = pack_set(full_space);

        Sequence!(Distribution!(State, Action))[] returnval = new Sequence!(Distribution!(State, Action))[trajectories.length];

        auto temporary = new Distribution!(State)(m.S(), DistInitType.Uniform);
        auto missing_observation = pack_distribution(policy * temporary);
                    
        foreach (i, traj ; trajectories) {
            
            // build an observation sequence

             Sequence!(Distribution!(Tuple!(State, Action))) observations = new Sequence!(Distribution!(Tuple!(State, Action)))(traj.length);
             foreach( t, timestep; traj) {

                Distribution!(Tuple!(State, Action)) dist;

                if (timestep[0] is null) {
                    // state is missing
                    dist = missing_observation;
                   
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
            
           foreach(repeat; 0 .. repeats) {
                
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


// alternative to my implementation using a markov smoother
class MarkovSmootherExactPartialTrajectoryToTrajectoryDistr: MCMCPartialTrajectoryToTrajectoryDistr {
    
    public this(Model m, LinearReward r, bool extend_terminals_to_equal_length = true) {
        super(m, r, 1, extend_terminals_to_equal_length);
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

class GibbsSamplingApproximatePartialTrajectoryToTrajectoryDistr: MCMCPartialTrajectoryToTrajectoryDistr {

    protected size_t burn_in_samples;
    protected size_t num_samples;
        
    public this(Model m, LinearReward r, size_t repeats, size_t burn_in_samples, size_t num_samples, bool extend_terminals_to_equal_length = true) {
        super(m, r, repeats, extend_terminals_to_equal_length);

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


enum HybridMCMCMode {

    Fixed,
    Adaptive,
    AdaptiveImportanceSampling
}

class HybridMCMCApproximatePartialTrajectoryToTrajectoryDistr: MCMCPartialTrajectoryToTrajectoryDistr {

    protected size_t burn_in_samples;
    protected size_t num_samples;
    protected Sequence!(Distribution!(Tuple!(State, Action)))[] proposalDistributions;
    protected HybridMCMCMode mode;
            
    public this(Model m, LinearReward r, size_t repeats, size_t burn_in_samples, size_t num_samples, Sequence!(Distribution!(State, Action))[] proposalDistributions, HybridMCMCMode mode = HybridMCMCMode.AdaptiveImportanceSampling, bool extend_terminals_to_equal_length = true) {
        super(m, r, repeats, extend_terminals_to_equal_length);

        this.burn_in_samples = burn_in_samples;
        this.num_samples = num_samples;
        this.mode = mode;

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

        switch (mode) {

            case HybridMCMCMode.Fixed:
                temp_sequence = HybridMCMC!(Tuple!(State, Action))(observations, transitions, initial_state, proposalDistributions[traj_num], burn_in_samples, num_samples);
                break;
            case HybridMCMCMode.Adaptive:
                temp_sequence = AdaptiveHybridMCMC!(Tuple!(State, Action))(observations, transitions, initial_state, proposalDistributions[traj_num], burn_in_samples, num_samples);
                break;
            case HybridMCMCMode.AdaptiveImportanceSampling:
                temp_sequence = AdaptiveHybridMCMCIS!(Tuple!(State, Action))(observations, transitions, initial_state, proposalDistributions[traj_num], burn_in_samples, num_samples);
                break;
            default:
                throw new Exception("Invalid Mode");
        }
        
        auto results = new Sequence!(Distribution!(State, Action))(temp_sequence.length);

        foreach (t, timestep; temp_sequence) {
            results[t] = tuple(unpack_distribution(timestep[0]));
        }

        return results;        

    }
}

// ******************************************************************************
// Partially observed trajectories
// ******************************************************************************

// Takes partially observed trajectories and converts them to trajectory distributions
// Exact version, interates through all possible trajectories using a forward-backword algorithm
class ExactPartiallyObservedTrajectoryToTrajectoryDistr : Sequence_Distribution_Computer!(State, Action, Observation) {

    protected bool extend_terminals_to_equal_length;
    protected Model m;
    protected LinearReward r;
    protected ConditionalDistribution!(Observation, State, Action) observation_model;
    protected Set!Observation observation_space;
    
    public this(Model m, LinearReward r, ConditionalDistribution!(Observation, State, Action) observation_model, Set!Observation observation_space, bool extend_terminals_to_equal_length = true) {
        this.extend_terminals_to_equal_length = extend_terminals_to_equal_length;
        this.m = m;
        this.r = r;
        this.observation_model = observation_model;
        this.observation_space = observation_space;
    }

    public Sequence!(Distribution!(State, Action, Observation))[] to_traj_distr( Sequence!(State, Action, Observation)[] trajectories, double [] weights ) {

        r.setWeights(weights);
        m.setR(r.toFunction());
        
        auto policy = m.getPolicy();

        auto full_space = m.S().cartesian_product(m.A()).cartesian_product(observation_space);        
        auto tuple_full_space = pack_set(full_space);
        auto stateaction_space = m.S().cartesian_product(m.A());
        auto tuple_stateaction_space = pack_set(stateaction_space);

        
        Sequence!(Distribution!(State, Action)) [] observation_array = new Sequence!(Distribution!(State, Action)) [trajectories.length]; 
        foreach (i, traj ; trajectories) {
            
            // build an observation sequence

             Sequence!(Distribution!(State, Action)) observations = new Sequence!(Distribution!(State, Action))(traj.length);
             Distribution!(State, Action) just_transitions;
             foreach( t, timestep; traj) {


                // build Pr(S, A) each timestep
                if (t == 0) {
                    just_transitions = policy * m.initialStateDistribution();
                } else {
                    just_transitions = policy * new Distribution!(State)(sumout!Action(sumout!State((m.T() * just_transitions).reverse_params())));
                }

                // multiply Pr( O | S, A) by Pr(S, A) to get Pr(S, A, O)

                Distribution!(State, Action, Observation) obs_prob = observation_model * just_transitions;
                
                // build Pr(S, A; O) by using the specific observation at this timestep                 
                Distribution!(State, Action) dist = new Distribution!(State, Action)(stateaction_space, 0.0);

                foreach( sa; stateaction_space) {
                    dist[sa[0], sa[1]] = obs_prob[ sa[0], sa[1], timestep[2] ];
                }

                dist.normalize();
                observations[t] = tuple(dist);
             }

            observation_array[i] = observations;
        }

        
        Sequence!(Distribution!(State, Action))[] forward = new Sequence!(Distribution!(State, Action))[trajectories.length];

        size_t max_traj_length = 0;
        foreach (t ; trajectories)
            if (t.length() > max_traj_length)
                max_traj_length = t.length();

        // forward step
        foreach(i, traj ; trajectories) {

            Sequence!(Distribution!(State, Action)) seq = new Sequence!(Distribution!(State, Action))(traj.length());
            foreach(t, timestep ; traj) {
                Distribution!(State, Action) dist;

                if (t == 0) {
                    // first timestep, use initial state distribution
                    dist = first_timestep(m.initialStateDistribution(), policy, observation_array[i][t][0], t);
                } else {
                    dist = forward_timestep(seq[t-1][0], policy, observation_array[i][t][0], t);
                }

                seq[t] = tuple(dist);
            }

            if (traj[$][0] ! is null && traj[$][0].isTerminal() && extend_terminals_to_equal_length ) {

                while( seq.length() < max_traj_length) {
                    seq ~= seq[$];
                }
            }
            
            forward[i] = seq;
        }
        
        Sequence!(Distribution!(State, Action))[] reverse = new Sequence!(Distribution!(State, Action))[trajectories.length];
        // reverse step
        foreach(i, traj ; trajectories) {
            Sequence!(Distribution!(State, Action)) seq = new Sequence!(Distribution!(State, Action))(traj.length());
            seq[$] = tuple(new Distribution!(State, Action)(stateaction_space, 1.0));
            
            foreach_reverse(t, timestep ; traj) {

                if (trajectories[i].length > t) {

                    Distribution!(State, Action,) dist;

                    if (t < traj.length - 1) {
                        dist = reverse_timestep(observation_array[i][t][0], seq[t+1][0], t);
                    } else {
                        dist = reverse_timestep(observation_array[i][t][0], new Distribution!(State, Action)(stateaction_space, 1.0), t);
                    }        

                    seq[t] = tuple(dist);
                }
                if (traj[$][0] ! is null && traj[$][0].isTerminal() && extend_terminals_to_equal_length ) {

                    while( seq.length() < max_traj_length) {
                        seq ~= seq[$];
                    }
                }
            }
            reverse[i] = seq;            
            
        }

        Sequence!(Distribution!(State, Action))[] combined = new Sequence!(Distribution!(State, Action))[trajectories.length];
        // combine together
        foreach(i, traj ; forward) {
            combined[i] = new Sequence!(Distribution!(State, Action))(traj.length());
            foreach( t, timestep; traj) {
                combined[i][t] = tuple(new Distribution!(State, Action)(timestep[0] * reverse[i][t][0]));
                combined[i][t][0].normalize();
            }
        }

        Sequence!(Distribution!(State, Action, Observation))[] returnval = new Sequence!(Distribution!(State, Action, Observation))[trajectories.length];

        foreach(i, temp_sequence; combined) {

            Sequence!(Distribution!(State, Action, Observation)) results = new Sequence!(Distribution!(State, Action, Observation))(temp_sequence.length);
            // convert results from Pr(S, A) => Pr(S, A, o)
            foreach (t, timestep; temp_sequence) {

                double[Tuple!(State, Action, Observation)] temp_array;
                foreach(sa; stateaction_space) {
                    temp_array[tuple(sa[0], sa[1], trajectories[i][t][2])] = timestep[0][sa[0], sa[1]];
                }
                Distribution!(State, Action, Observation) temp_distribution = new Distribution!(State, Action, Observation)(full_space, temp_array);
                results[t] = tuple(temp_distribution);
            }
            returnval[i] = results;
        }
                
        return returnval;

    }

    protected Distribution!(State, Action) first_timestep(Distribution!(State) initial, ConditionalDistribution!(Action, State) policy, Distribution!(State, Action) observation, size_t timestep) {

        auto returnval = new Distribution!(State, Action)(observation * (policy * initial));
//        returnval.normalize();        
        return returnval;

    }
    
    protected Distribution!(State, Action) forward_timestep(Distribution!(State, Action) previous_timestep, ConditionalDistribution!(Action, State) policy, Distribution!(State, Action) observation,  size_t timestep) {

        auto returnval = new Distribution!(State, Action)(observation * (policy * new Distribution!(State)(sumout!(Action)(sumout!(State)( (m.T() * previous_timestep).reverse_params())))));
//        returnval.normalize();        
        return returnval;

    }

    protected Distribution!(State, Action) reverse_timestep(Distribution!(State, Action) observation, Distribution!(State, Action) next_timestep, size_t timestep) {
        auto returnval = new Distribution!(State, Action)(sumout!(State)( ((m.T() * observation) * sumout!(Action)(next_timestep ) ) ) );
//        returnval.normalize();
        return returnval;
    }
}


// alternative to my implementation using a markov smoother
class MarkovSmootherExactPartiallyObservedTrajectoryToTrajectoryDistr: Sequence_Distribution_Computer!(State, Action, Observation) {

    bool extend_terminals_to_equal_length;
    Model m;
    LinearReward r;
    ConditionalDistribution!(Observation, State, Action) observation_model;
    Set!Observation observation_space;
    
    public this(Model m, LinearReward r, ConditionalDistribution!(Observation, State, Action) observation_model, Set!Observation observation_space, bool extend_terminals_to_equal_length = true) {
        this.extend_terminals_to_equal_length = extend_terminals_to_equal_length;
        this.m = m;
        this.r = r;
        this.observation_model = observation_model;
        this.observation_space = observation_space;
    }

    public Sequence!(Distribution!(State, Action, Observation))[] to_traj_distr( Sequence!(State, Action, Observation)[] trajectories, double [] weights ) {

        r.setWeights(weights);
        m.setR(r.toFunction());
        
        auto policy = m.getPolicy();

        auto full_space = m.S().cartesian_product(m.A()).cartesian_product(observation_space);        
        auto tuple_full_space = pack_set(full_space);
        auto stateaction_space = m.S().cartesian_product(m.A());
        auto tuple_stateaction_space = pack_set(stateaction_space);
        
        Sequence!(Distribution!(State, Action, Observation))[] returnval = new Sequence!(Distribution!(State, Action, Observation))[trajectories.length];

        foreach (i, traj ; trajectories) {
            
            // build an observation sequence

             Sequence!(Distribution!(Tuple!(State, Action))) observations = new Sequence!(Distribution!(Tuple!(State, Action)))(traj.length);
             Distribution!(State, Action) just_transitions;
             foreach( t, timestep; traj) {


                // build Pr(S, A) each timestep
                if (t == 0) {
                    just_transitions = policy * m.initialStateDistribution();
                } else {
                    just_transitions = policy * new Distribution!(State)(sumout!Action(sumout!State((m.T() * just_transitions).reverse_params())));
                }

                // multiply Pr( O | S, A) by Pr(S, A) to get Pr(S, A, O)

                Distribution!(State, Action, Observation) obs_prob = observation_model * just_transitions;
                
                // build Pr(S, A; O) by using the specific observation at this timestep                 
                Distribution!(Tuple!(State, Action)) dist = new Distribution!(Tuple!(State, Action))(tuple_stateaction_space, 0.0);

                foreach( sa; tuple_stateaction_space) {
                    dist[sa[0]] = obs_prob[ sa[0][0], sa[0][1], timestep[2] ];
                }

                dist.normalize();
                observations[t] = tuple(dist);
             }

             
            // build a transition function
            
            auto transitions = new ConditionalDistribution!(Tuple!(State, Action), Tuple!(State, Action))(tuple_stateaction_space, tuple_stateaction_space);
            foreach (sa ; stateaction_space) {
                transitions[sa] = pack_distribution( policy * m.T()[sa] );
                
            }

            Distribution!(Tuple!(State, Action)) initial_state = pack_distribution(policy * m.initialStateDistribution());
            
            auto temp_sequence = SequenceMarkovChainSmoother!(Tuple!(State, Action))(observations, transitions, initial_state);
            auto results = new Sequence!(Distribution!(State, Action, Observation))(temp_sequence.length);

            // convert results from Pr(S, A) => Pr(S, A, o)
            foreach (t, timestep; temp_sequence) {

                double[Tuple!(State, Action, Observation)] temp_array;
                foreach(sa; stateaction_space) {
                    temp_array[tuple(sa[0], sa[1], traj[t][2])] = timestep[0][sa];
                }
                Distribution!(State, Action, Observation) temp_distribution = new Distribution!(State, Action, Observation)(full_space, temp_array);
                results[t] = tuple(temp_distribution);
            }
            returnval[i] = results;
        }

        return returnval;                
    }

}


