module occlusion;

import discretemdp;
import discretefunctions;
import trajectories;
import std.typecons;
import std.algorithm.comparison;
import std.random;


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


// alternative to my implementation using a markov smoother
class MarkovSmootherExactOccludedTrajectoryToTrajectoryDistr: Sequence_Distribution_Computer!(State, Action) {

    bool extend_terminals_to_equal_length;
    Model m;
    LinearReward r;
    Set!State occluded_states;
    
    public this(Model m, LinearReward r, Set!State occluded_states, bool extend_terminals_to_equal_length = true) {
        this.extend_terminals_to_equal_length = extend_terminals_to_equal_length;
        this.m = m;
        this.r = r;
        this.occluded_states = occluded_states;
    }

    public Sequence!(Distribution!(State, Action))[] to_traj_distr( Sequence!(State, Action)[] trajectories, double [] weights ) {

        r.setWeights(weights);
        m.setR(r.toFunction());
        
        auto policy = m.getPolicy();

        auto full_space = m.S().cartesian_product(m.A());        
        auto tuple_full_space = pack_set(full_space);

        Sequence!(Distribution!(State, Action))[] returnval = new Sequence!(Distribution!(State, Action))[trajectories.length];

        auto temporary = new Distribution!(State)(m.S(), 0.0);
        foreach (s; occluded_states) {
            // only the occluded states could be in the missing timesteps
            temporary[s] = 1.0 / occluded_states.size();
        }
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
            
            auto temp_sequence = SequenceMarkovChainSmoother!(Tuple!(State, Action))(observations, transitions, initial_state);
            auto results = new Sequence!(Distribution!(State, Action))(temp_sequence.length);

            foreach (t, timestep; temp_sequence) {
                results[t] = tuple(unpack_distribution(timestep[0]));
            }
            returnval[i] = results;
        }

        return returnval;                
    }

}
