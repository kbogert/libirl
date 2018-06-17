module maxentIRL;

import discretemdp;
import discretefunctions;
import solvers;
import std.random;
import std.typecons;
import utility;
import std.math;
import std.array;

/*

 1. Write test cases
 2. Design interface
 3. Verify against toy problem
 4. Implement MaxCausalEntropy

 5. Implement Original and improved versions of mIRL*
 6. Implement HiddenDataEM
 7. Optimize HiddenDataEM for robots
 
*/

// Deterministic MDPs only

class MaxEntIRL_exact {

    protected Model model;
    protected double tol;
    protected LinearReward reward;
    protected size_t max_traj_length;
    
    public this (Model m, LinearReward lw, double tolerance) {
        model = m;
        reward = lw;
        tol = tolerance;

        // verify that the MDP is deterministic

        foreach (s; m.S()) {
            foreach(a ; m.A()) {
                foreach (s_prime ; m.S()) {
                    auto prob = m.T()[tuple(s[0], a[0])][s_prime];
                    if (prob < 1.0 && prob > 0.0)
                        throw new Exception("MaxEntIRL only works with deterministic MDPs");
                }
            }
        }
    }


    public double [] solve (Sequence!(State, Action)[] trajectories, bool stochasticGradientDescent = true) {

        double [] returnval = minimallyInitializedArray!(double[])(reward.getSize());
        foreach (i ; 0 .. returnval.length)
            returnval[i] = uniform(0.0, 1.0);

        max_traj_length = 0;
        foreach (t ; trajectories)
            if (t.length() > max_traj_length)
                max_traj_length = t.length();

        // convert trajectories into an array of features

        auto expert_fe = feature_expectations_from_trajectories(trajectories, reward);

        if (stochasticGradientDescent)
            returnval = unconstrainedAdaptiveExponentiatedStochasticGradientDescent(expert_fe, 1.0, tol, size_t.max, & EEFFeaturesAtTimestep);
        else {
            // normalize initial weights
            returnval[] /= l1norm(returnval);

            double [] avg_fe = minimallyInitializedArray!(double [])(reward.getSize());
 
            foreach (fe ; expert_fe) {
                avg_fe[] += fe[] / expert_fe.length;
//import std.stdio;
//writeln(avg_fe);
            }
            returnval = exponentiatedGradientDescent(avg_fe, returnval.dup, 5.0, tol, size_t.max, max_traj_length, & EEFFeatures);
        }            
        return returnval;
    }

    Function!(double, State, size_t) ExpectedEdgeFrequency(double [] weights, size_t N, out double [Action][State] P_a_s) {

        Function!(double, State) Z_s = new Function!(double, State)(model.S(), 0.0);
        Function!(double, State, Action) Z_a = new Function!(double, State, Action)(model.S().cartesian_product(model.A()), 0.0);

        auto T = model.T().flatten();

        // terminals MUST be infinite!
        
        foreach(s; model.S()) {
            if (s[0].isTerminal()) {
                Z_s[s[0]] = 1.0;

                foreach (s_p; model.S()) {

                    foreach (a; model.A()) {

                        if (s_p == s) {
                            T[tuple(s[0], a[0], s_p[0])] = 1.0;
                        } else {
                            T[tuple(s[0], a[0], s_p[0])] = 0.0;
                        }

                    }
                }
                
            }
        }

        Function!(double, State) terminals = new Function!(double, State)(Z_s);

        reward.setWeights(weights);
        Function!(double, State, Action) exp_reward = reward.toFunction();
        foreach (s; model.S()) {
            foreach(a; model.A()) {
                exp_reward[s[0],a[0]] = exp(exp_reward[s[0],a[0]]);
            }
        }

        auto state_reward = max(exp_reward);

        Z_s = Z_s * state_reward;
        
import std.stdio;
//writeln(exp_reward);
        foreach (n ; 0 .. N) {

            Z_a = sumout( T * Z_s ) * exp_reward;
            Z_s = sumout( Z_a ) + terminals * state_reward;
//writeln(Z_a);
//writeln();
        }


        // Need to build a function (action, state) reduces Z_a, loop for each action, and save the results to P
        // awkward ...

        foreach (a; model.A()) {
            auto choose_a = new Function!(Tuple!(Action), State)(model.S(), a);
            auto temp = Z_a.apply(choose_a) / Z_s;

            foreach (s; model.S()) {
                P_a_s[s[0]][a[0]] = temp[s[0]];
            }
        }

//        writeln(P_a_s);

        NumericSetSpace num_set = new NumericSetSpace(N);

        auto D_s_t = new Function!(double, State, size_t)(model.S().cartesian_product(num_set), 0.0);

        foreach (s; model.S()) {
            D_s_t[tuple(s[0], cast(size_t)0)] = model.initialStateDistribution()[s[0]];
        }

        foreach (t; 1 .. N) {
            foreach (k; model.S()) {
                double temp = 0.0;
                foreach (i; model.S()) {
                    double val = D_s_t[tuple(i[0], (t-1))];
//                    if (i[0].isTerminal()) {
//                        if (i == k)
//                            temp += val; 
//                    } else {
                        foreach (j; model.A()) {
                            temp += val * P_a_s[i[0]][j[0]] * T[tuple(i[0], j[0], k[0])];
                        }
//                    }
                }
                D_s_t[tuple(k[0], t)] = temp;
            }
        }
        
        return D_s_t;
    }

    double [] EEFFeatures(double [] weights) {

        double [Action][State] P_a_s;
        
        auto D = ExpectedEdgeFrequency(weights, max_traj_length, P_a_s);
        auto Ds = sumout(D);

//import std.stdio;
//writeln(D);        
        double [] returnval = minimallyInitializedArray!(double[])(reward.getSize());
        
        foreach (s; Ds.param_set()) {

  /*          if (s[0].isTerminal()) {
                    // just use the first action to get the features for the terminal state
                    auto features = reward.getFeatures(s[0], model.A().toArray()[0][0]);
                    returnval[] += Ds[s[0]] * features[];
            } else {
  */              foreach (a; model.A() ) {
                    auto features = reward.getFeatures(s[0], a[0]);

                    returnval[] += Ds[s[0]] * P_a_s[s[0]][a[0]] * features[];
                }
//            }
        }


        return returnval;

    }    

    double [] EEFFeaturesAtTimestep(double [] weights, size_t timestep) {

        double [Action][State] P_a_s;

        auto D = ExpectedEdgeFrequency(weights, max_traj_length, P_a_s);

        double [] returnval = minimallyInitializedArray!(double[])(reward.getSize());

        foreach (s; model.S()) {

/*            if (s[0].isTerminal()) {
                    // just use the first action to get the features for the terminal state
                    auto features = reward.getFeatures(s[0], model.A().toArray()[0][0]);
                    returnval[] += D[tuple(s[0], timestep)] * features[];
            } else {
*/                foreach (a; model.A() ) {
                    auto features = reward.getFeatures(s[0], a[0]);

                    returnval[] += D[tuple(s[0], timestep)] * P_a_s[s[0]][a[0]] * features[];

                }
//            }    
        }


        return returnval;

    }    

}


double [][] feature_expectations_from_trajectories(Sequence!(State, Action)[] trajectories, LinearReward reward) {

    double [][] returnval;

    foreach (t; trajectories) {
        returnval ~= feature_expectations_from_trajectory(t, reward);
    }
//    import std.stdio;
//    writeln(returnval);
    return returnval;
}

double [] feature_expectations_from_trajectory(Sequence!(State, Action) trajectory, LinearReward reward) {

    double [] returnval = minimallyInitializedArray!(double[])(reward.getSize());

    foreach(sa; trajectory) {
        if (sa[0].isTerminal() && sa[1] is null) {
            // the action on the terminal state is null, we're in trouble because the rewards
            // are defined for state/actions.  We'll just pick any action from the set and
            // use it, since if the features are defined correctly the action shouldn't
            // matter for terminal states.

            auto randomAction = reward.toFunction().param_set().toArray()[0][1];
            returnval[] += reward.getFeatures(sa[0], randomAction)[];
        } else {
            returnval[] += reward.getFeatures(sa[0], sa[1])[];
        }
    }
    
    return returnval;
    
}
