module featureexpectations;

import discretefunctions;
import discretemdp;
import std.typecons;



// Deterministic trajectories, such as those generated from simulate()

double [] feature_expectations_from_trajectories(Sequence!(State, Action)[] trajectories, LinearReward reward, size_t normalize_length_to = 0) {

    double [] returnval = new double[reward.getSize()];
    returnval[] = 0;

    foreach (t; trajectories) {
        returnval[] += feature_expectations_from_trajectory(t, reward, normalize_length_to)[] / trajectories.length;
    }

    return returnval;
}

double [] feature_expectations_from_trajectory(Sequence!(State, Action) trajectory, LinearReward reward, size_t normalize_length_to = 0) {

    double [] returnval = new double[reward.getSize()];
    returnval[] = 0;

    foreach(sa; trajectory) {
        if (sa[0].isTerminal() && sa[1] is null) {
            // the action on the terminal state is null, we're in trouble because the rewards
            // are defined for state/actions.  We'll just pick any action from the set and
            // use it, since if the features are defined correctly the action shouldn't
            // matter for terminal states.

            auto randomAction = reward.toFunction().param_set().getOne()[1];
            returnval[] += reward.getFeatures(sa[0], randomAction)[];
        } else {
            returnval[] += reward.getFeatures(sa[0], sa[1])[];
        }
    }

    if (trajectory.length() < normalize_length_to) {
        if (trajectory[$][0].isTerminal()) {
            // project the last state outwards to the desired trajectory length

            if (trajectory[$][1] is null) {
                auto randomAction = reward.toFunction().param_set().getOne()[1];
                returnval[] += (normalize_length_to - trajectory.length()) * reward.getFeatures(trajectory[$][0], randomAction)[];
            } else {
                returnval[] += (normalize_length_to - trajectory.length()) * reward.getFeatures(trajectory[$][0], trajectory[$][1])[];
            }            

        }
    }
    
    return returnval;
    
}

double [][] feature_expectations_per_timestep(Sequence!(State, Action)[] trajectories, LinearReward reward) {

    double [][] returnval;

    while(true) {

        size_t trajectories_found = 0;
        double [] next_timestep = new double[reward.getSize()];
        next_timestep[] = 0;

        auto t = returnval.length;
        foreach(traj ; trajectories) {
            if (traj.length > t) {
                trajectories_found ++;

                auto sa = traj[t];
                if (sa[0].isTerminal() && sa[1] is null) {
                    auto randomAction = reward.toFunction().param_set().getOne()[1];
                    next_timestep[] += reward.getFeatures(sa[0], randomAction)[];
                } else
                    next_timestep[] += reward.getFeatures(sa[0], sa[1])[];
            } else {
                // repeat the last entry in this trajectory to turn this into an infinite terminal
                auto sa = traj[$];
                if (sa[0].isTerminal() ) {
                    if (sa[1] is null) {
                        auto randomAction = reward.toFunction().param_set().getOne()[1];
                        next_timestep[] += reward.getFeatures(sa[0], randomAction)[];
                    } else
                        next_timestep[] += reward.getFeatures(sa[0], sa[1])[];

                }
            }
        }
        
        if (trajectories_found == 0)
            break;

        next_timestep[] /= trajectories.length;
        returnval ~= next_timestep;
        
    }


    return returnval;
}


// non-deterministic trajectories, such as those from observations


double [] feature_expectations_from_trajectories(T ...)(Sequence!(Distribution!(T))[] trajectories, double [] delegate(T) feature_function, size_t ff_size) {

    double [] returnval = new double[ff_size];
    returnval[] = 0;

    foreach (t; trajectories) {
        returnval[] += feature_expectations_from_trajectory(t, feature_function, ff_size)[] / trajectories.length;
    }

    return returnval;
}

double [] feature_expectations_from_trajectory(T ...)(Sequence!(Distribution!(T)) trajectory, double [] delegate(T) feature_function, size_t ff_size) {

    double [] returnval = new double[ff_size];
    returnval[] = 0;

    foreach(timestep; trajectory) {
        foreach(nodes; timestep[0].param_set()) {
            auto prob = timestep[0][nodes];
            if (prob > 0)
                returnval[] += feature_function(nodes.expand)[] * prob;
        }
    }
    
    return returnval;
    
}

double [][] feature_expectations_per_timestep(T ...)(Sequence!(Distribution!(State, Action))[] trajectories, double [] delegate(T) feature_function, size_t ff_size) {

    double [][] returnval;

    while(true) {

        size_t trajectories_found = 0;
        double [] next_timestep = new double[ff_size];
        next_timestep[] = 0;

        auto t = returnval.length;
        foreach(traj ; trajectories) {
            if (traj.length > t) {
                trajectories_found ++;

                foreach(param; traj[t][0].param_set()) {
                    auto prob = traj[t][0][param];
                    if (prob > 0)
                        next_timestep[] += feature_function(param.expand)[] * prob;
                }
            } 
        }
        
        if (trajectories_found == 0)
            break;

        next_timestep[] /= trajectories_found;
        returnval ~= next_timestep;
        
    }


    return returnval;
}

