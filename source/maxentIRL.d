module maxentIRL;

import discretemdp;
import discretefunctions;
import solvers;
import std.random;
import std.typecons;
import utility;
import std.math;
import std.array;
import std.algorithm;
import std.numeric;
import std.variant;
import featureexpectations;
import trajectories;

/*

 1. Write test cases
 2. Design interface
 3. Verify against toy problem
 4. Implement MaxCausalEntropy

 5. Implement Original and improved versions of mIRL*
 6. Implement HiddenDataEM
 7. Optimize HiddenDataEM for robots
 
*/

interface Sequence_MaxEnt_Problem(T ...) {
    
    public double [] solve (Sequence!(Distribution!(T)) [] trajectories);

}

// Implementation of Ziebart 2008 (Expected Edge Frequency Algorithm)
// Allows for non-deterministic MDPS, assuming the stochasticity is low and unimportant
// YMMV
class MaxEntIRL_Ziebart_approx : Sequence_MaxEnt_Problem!(State, Action) {

    protected Model model;
    protected double tol;
    protected LinearReward reward;
    protected size_t max_traj_length;

    protected size_t sgd_block_size;
    protected size_t inference_counter;
    protected Variant inference_cache;
    protected bool stochasticGradientDescent;
        
    protected double [] true_weights;
    public this (Model m, LinearReward lw, double tolerance, double [] true_weights, bool stochasticGradientDescent = true) {
        model = m;
        reward = lw;
        tol = tolerance;
        this.true_weights = true_weights;
        sgd_block_size = 1;
        inference_counter = 0;
        this.stochasticGradientDescent = stochasticGradientDescent;
    }


    public double [] solve (Sequence!(Distribution!(State, Action))[] trajectories) {

        double [] returnval = new double[reward.getSize()];
        foreach (i ; 0 .. returnval.length)
            returnval[i] = uniform(0.0, 1.0);

        max_traj_length = 0;
        foreach (t ; trajectories)
            if (t.length() > max_traj_length)
                max_traj_length = t.length();

        // convert trajectories into an array of features

        if (stochasticGradientDescent) {

            auto expert_fe = feature_expectations_per_timestep(trajectories, &reward.getFeatures, reward.getSize());
            
//            sgd_block_size = max(3, max_traj_length / 4);
            sgd_block_size = 1;
            inference_counter = 0;
                        
            returnval = unconstrainedAdaptiveExponentiatedStochasticGradientDescent(expert_fe, 1, tol, 1000, & EEFFeaturesAtTimestep, true);
        } else {
            // normalize initial weights
            returnval[] /= l1norm(returnval);

            auto expert_fe = feature_expectations_from_trajectories(trajectories, &reward.getFeatures, reward.getSize());

            returnval = exponentiatedGradientDescent(expert_fe, returnval.dup, 2.0, tol, size_t.max, max_traj_length, & EEFFeatures);
        }            
        return returnval;
    }

    Function!(double, State, size_t) ExpectedEdgeFrequency(double [] weights, size_t N, size_t D_length, out double [Action][State] P_a_s) {

//        weights = true_weights.dup;

            // terminals MUST be infinite!

        Function!(double, State) Z_s = new Function!(double, State)(model.S(), 0.0);
        auto T = model.T().flatten();
        
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

        if ( inference_counter != 0 && inference_cache.hasValue()) {
            // use the cached P_a_s instead of calculating it

            P_a_s = inference_cache.get!(double[Action][State])();
            
        } else {

            weights = utility.clamp(weights, -25, 25);
        
            Function!(double, State, Action) Z_a = new Function!(double, State, Action)(model.S().cartesian_product(model.A()), 0.0);

            Function!(double, State) terminals = new Function!(double, State)(Z_s);

            reward.setWeights(weights);
            Function!(double, State, Action) exp_reward = reward.toFunction();
            foreach (s; model.S()) {
                foreach(a; model.A()) {
                    exp_reward[s[0],a[0]] = exp(exp_reward[s[0],a[0]]);
                }
            }

    //import std.stdio;        
    //writeln(exp_reward);
    //        auto state_reward = max(exp_reward);

    //        Z_s = Z_s * state_reward;
        

    //writeln(exp_reward);
            foreach (n ; 0 .. N) {

                Z_a = sumout( T * Z_s ) * exp_reward;
                Z_s = sumout( Z_a ) + terminals /* state_reward*/;
    //writeln(Z_a);
    //writeln();
            }


            // Need to build a function (action, state) reduces Z_a, loop for each action, and save the results to P
            // awkward ...

            foreach (a; model.A()) {
    //            auto choose_a = new Function!(Tuple!(Action), State)(model.S(), a);
    //            auto temp = Z_a.apply(choose_a) / Z_s;

                foreach (s; model.S()) {
    //                P_a_s[s[0]][a[0]] = temp[s[0]];
                    P_a_s[s[0]][a[0]] = Z_a[tuple(s[0], a[0])] / Z_s[s[0]];
                }
            }

            inference_cache = P_a_s;
        }
        
        inference_counter = (inference_counter + 1) % sgd_block_size;

//        writeln(P_a_s);

        NumericSetSpace num_set = new NumericSetSpace(D_length);

        auto D_s_t = new Function!(double, State, size_t)(model.S().cartesian_product(num_set), 0.0);

        foreach (s; model.S()) {
            D_s_t[tuple(s[0], cast(size_t)0)] = model.initialStateDistribution()[s[0]];
        }
        

        foreach (t; 1 .. D_length) {
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
//writeln(D_s_t);        
        return D_s_t;
    }

    double [] EEFFeatures(double [] weights) {

        double [Action][State] P_a_s;
        
        auto D = ExpectedEdgeFrequency(weights, model.S().size(), max_traj_length, P_a_s);
        auto Ds = sumout(D);

//import std.stdio;
//writeln(D);
//writeln(Ds);        

        double [] returnval = new double[reward.getSize()];
        returnval[] = 0;
                
        foreach (s; Ds.param_set()) {

  /*          if (s[0].isTerminal()) {
                    // just use the first action to get the features for the terminal state
                    auto features = reward.getFeatures(s[0], model.A().getOne()[0]);
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

        auto D = ExpectedEdgeFrequency(weights, model.S().size(), max_traj_length, P_a_s);
        
        double [] returnval = new double[reward.getSize()];
        returnval[] = 0;

        foreach (s; model.S()) {

/*            if (s[0].isTerminal()) {
                    // just use the first action to get the features for the terminal state
                    auto features = reward.getFeatures(s[0], model.A().getOne()[0]);
                    returnval[] += D[tuple(s[0], timestep)] * features[];
            } else {
*/                foreach (a; model.A() ) {
//                    import std.stdio;
//                    writeln(s[0], " ", a[0], " ", timestep);
 //                   writeln(D);
                    auto features = reward.getFeatures(s[0], a[0]);

                    returnval[] += D[tuple(s[0], timestep)] * P_a_s[s[0]][a[0]] * features[];

                }
//            }    
        }

        return returnval;

    }    

}

// Implementation of Ziebart 2008 (Expected Edge Frequency Algorithm)
// Deterministic MDPs only
class MaxEntIRL_Ziebart_exact : MaxEntIRL_Ziebart_approx {

    public this (Model m, LinearReward lw, double tolerance, double [] true_weights, bool stochasticGradientDescent = true) {
        super(m, lw, tolerance, true_weights, stochasticGradientDescent);
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
}


// Implementation of Ziebart 2010

class MaxCausalEntIRL_Ziebart : Sequence_MaxEnt_Problem!(State, Action)  {

    protected Model model;
    protected double tol;
    protected LinearReward reward;
    protected size_t max_traj_length;

    protected size_t sgd_block_size;
    protected size_t inference_counter;
    protected Variant inference_cache;
    protected bool stochasticGradientDescent;
        
    protected double [] true_weights;
    public this (Model m, LinearReward lw, double tolerance, double [] true_weights, bool stochasticGradientDescent = true) {
        model = m;
        reward = lw;
        tol = tolerance;
        this.true_weights = true_weights;
        sgd_block_size = 1;
        inference_counter = 0;
        this.stochasticGradientDescent = stochasticGradientDescent;
    }


    public double [] solve (Sequence!(Distribution!(State, Action)) [] trajectories) {

        double [] returnval = new double[reward.getSize()];
        foreach (i ; 0 .. returnval.length)
            returnval[i] = uniform(0.0, 1.0);

        max_traj_length = 0;
        foreach (t ; trajectories)
            if (t.length() > max_traj_length)
                max_traj_length = t.length();

        // convert trajectories into an array of features
/*
import std.stdio;
writeln();
writeln("Empirical Feature expectations");
writeln(feature_expectations_from_trajectories(trajectories, &reward.getFeatures, reward.getSize()));
*/

        
        if (stochasticGradientDescent) {

            auto expert_fe = feature_expectations_per_timestep(trajectories, &reward.getFeatures, reward.getSize());
            
//            sgd_block_size = max(3, max_traj_length / 4);
            sgd_block_size = 1;
            inference_counter = 0;
                        
            returnval = unconstrainedAdaptiveExponentiatedStochasticGradientDescent(expert_fe, 1, tol, 1000, & GradientForTimestep, true);
        } else {

            auto expert_fe = feature_expectations_from_trajectories(trajectories, &reward.getFeatures, reward.getSize());

            returnval = unconstrainedAdaptiveExponentiatedGradientDescent(expert_fe, 0.28, tol, 50, & Gradient, true);
//            // normalize initial weights
//            returnval[] /= l1norm(returnval);
//            returnval = exponentiatedGradientDescent(expert_fe, returnval.dup, 2.0, tol, size_t.max, max_traj_length, & Gradient);
        }            
        return returnval;
    }

    ConditionalDistribution!(Action, State) [] inferenceProcedure (double [] weights, size_t T) {

        scope (exit)  inference_counter = (inference_counter + 1) % sgd_block_size;
        
        if ( inference_counter != 0 && inference_cache.hasValue()) {
            // use the cached P(A|S) instead of calculating it

            return inference_cache.get!(ConditionalDistribution!(Action, State)[])();
            
        } else {
                    
            auto transitions = model.T().flatten();
            auto log_Z_s = new Function!(double, State)(model.S(), 0.0);
            ConditionalDistribution!(Action, State) [] P_a_s_t;
            P_a_s_t.length = T;
        
            foreach_reverse (t; 0 .. T) {

                auto log_Z_a_s = new Function!(double, State, Action)(model.S().cartesian_product(model.A()), 0.0);
                foreach (s ; model.S()) {
                    foreach(a ; model.A()) {
                        log_Z_a_s[s[0], a[0]] = dotProduct(weights, reward.getFeatures(s[0], a[0]));
                        if (t < T) {
                            foreach (s_p; model.S()) {
                                log_Z_a_s[s[0], a[0]] += transitions[s[0], a[0], s_p[0]] * log_Z_s[s_p[0]];
                            }
                        }
                    }
                }

                log_Z_s = softmax(log_Z_a_s);

                double [Tuple!Action][Tuple!State] P_a_s;
                foreach (s ; model.S()) {
                    foreach(a ; model.A()) {
                        P_a_s[s][a] = exp(log_Z_a_s[s[0], a[0]]) / exp(log_Z_s[s[0]]); 
                    }
                }
                        
                P_a_s_t[t] = new ConditionalDistribution!(Action, State)(model.A(), model.S(), P_a_s);
            }


            inference_cache = P_a_s_t;
            
            return P_a_s_t;
        }
    }

    Distribution!(State, Action)[] StateActionDistributionPerTimestep(ConditionalDistribution!(Action, State)[] policy) {

        auto transitions = model.T().flatten();

        Distribution!(State, Action)[] D_s_a_t;

import std.stdio;
//writeln(policy[$-1]);
//writeln();
        foreach (t; 0 .. policy.length) {

            auto D_s_a = new Distribution!(State, Action)(model.S().cartesian_product(model.A()), 0.0);
            
            foreach (s; model.S()) {
                foreach (a; model.A()) {
                    if (t == 0) {
                        D_s_a[s[0], a[0]] = model.initialStateDistribution()[s[0]] * policy[t][s][a];
                    } else {

                        foreach(s_p; model.S()) {
                            foreach (a_p; model.A()) {
                                D_s_a[s[0], a[0]] += D_s_a_t[t-1][s_p[0], a_p[0]] * transitions[s_p[0], a_p[0], s[0]] * policy[t][s][a];
                            }
                        }
                        
                    }

                }

            }
//writeln(D_s_a);
//writeln();
            D_s_a_t ~= D_s_a;
        }

        return D_s_a_t;
    }
    
    double [] Gradient(double [] weights) {

        
        auto D_s_a = StateActionDistributionPerTimestep(inferenceProcedure(weights, max_traj_length));

        double [] returnval = new double[reward.getSize()];
        returnval[] = 0;
                
        foreach (D; D_s_a) {
            foreach (s ; model.S()) {
                foreach(a ; model.A()) {
                    returnval[] += D[s[0], a[0]] * reward.getFeatures(s[0], a[0])[];
                }
            }
        }
    
        return returnval;

    }    

    double [] GradientForTimestep(double [] weights, size_t timestep) {

        auto D_s_a = StateActionDistributionPerTimestep(inferenceProcedure(weights, max_traj_length));

//import std.stdio;
//writeln(sgd_callback_cache[timestep]);
        double [] returnval = new double[reward.getSize()];
        returnval[] = 0;

        foreach (s; model.S()) {
            foreach (a; model.A() ) {
                returnval[] += D_s_a[timestep][s[0], a[0]] * reward.getFeatures(s[0], a[0])[];

            }
        }

        return returnval;

    }    


}


class MaxCausalEntIRL_InfMDP : MaxCausalEntIRL_Ziebart {
    
    public this (Model m, LinearReward lw, double tolerance, double [] true_weights, bool stochasticGradientDescent = true) {
        super(m, lw, tolerance, true_weights, stochasticGradientDescent);
    }
    
    override ConditionalDistribution!(Action, State) [] inferenceProcedure (double [] weights, size_t T) {

        reward.setWeights(weights);
        model.setR(reward.toFunction());

        auto V_soft = soft_max_value_iteration(model, model.getValueIterationTolerance(), inference_cache);
        auto policy = soft_max_policy(V_soft, model);

        ConditionalDistribution!(Action, State) [] returnval = new ConditionalDistribution!(Action, State) [T];

        foreach (i; 0 .. T) {
            returnval[i] = policy;
        }

        inference_cache = V_soft;
        return returnval;
    }
}

class MaxCausalEntIRL_SGDApprox : MaxCausalEntIRL_InfMDP {

    protected Distribution!(State, Action) [] empirical_D_s_a_t;
    
    public this (Model m, LinearReward lw, double tolerance, double [] true_weights, bool stochasticGradientDescent = true) {
        super(m, lw, tolerance, true_weights, stochasticGradientDescent);
    }

    override public double [] solve (Sequence!(Distribution!(State, Action)) [] trajectories) {

        max_traj_length = 0;
        foreach (t ; trajectories)
            if (t.length() > max_traj_length)
                max_traj_length = t.length();                

        auto sa_set = model.S().cartesian_product(model.A());
        
        empirical_D_s_a_t = new Distribution!(State, Action) [max_traj_length];
        foreach(t; 0 .. max_traj_length) {
            empirical_D_s_a_t[t] = new Distribution!(State, Action) (sa_set, 0.0);
            
            foreach(traj; trajectories) {
                if (traj.length > t) {
                    foreach (sa; traj[t][0].param_set())
                        empirical_D_s_a_t[t][sa] += traj[t][0][sa];
                }
            }
            empirical_D_s_a_t[t].normalize();

        }
        return super.solve(trajectories);

    }

    
    Distribution!(State, Action) StateActionDistributionAtTimestep(ConditionalDistribution!(Action, State)[] policy, size_t timestep) {
        auto D_s_a = new Distribution!(State, Action)(model.S().cartesian_product(model.A()), 0.0);

        auto transitions = model.T().flatten();
  
        foreach (s; model.S()) {
            foreach (a; model.A()) {
                if (timestep == 0) {
                    D_s_a[s[0], a[0]] = model.initialStateDistribution()[s[0]] * policy[timestep][s][a];
                } else {

                    foreach(s_p; model.S()) {
                        double pr_s_t = 0.0;
                        foreach (a_p_p; model.A()) {
                            pr_s_t += empirical_D_s_a_t[timestep-1][s_p[0], a_p_p[0]];              
                        }
                        foreach (a_p; model.A()) {                            
                            D_s_a[s[0], a[0]] += pr_s_t * policy[timestep-1][s_p][a_p] * transitions[s_p[0], a_p[0], s[0]] * policy[timestep][s][a];
                        }
                    }   
                }
            }
        }

        return D_s_a;        
    }

    override double [] GradientForTimestep(double [] weights, size_t timestep) {

        auto P_a_s = StateActionDistributionAtTimestep(inferenceProcedure(weights, max_traj_length), timestep);

        double [] returnval = new double[reward.getSize()];
        returnval[] = 0;

        foreach (s; model.S()) {
            foreach (a; model.A() ) {
                returnval[] += P_a_s[s[0], a[0]] * reward.getFeatures(s[0], a[0])[];

            }
        }

        return returnval;        
    }
}

class MaxCausalEntIRL_SGDEmpirical : MaxCausalEntIRL_SGDApprox {

    public this (Model m, LinearReward lw, double tolerance, double [] true_weights, bool stochasticGradientDescent = true) {
        super(m, lw, tolerance, true_weights, stochasticGradientDescent);
    }
    
    override Distribution!(State, Action) StateActionDistributionAtTimestep(ConditionalDistribution!(Action, State)[] policy, size_t timestep) {
        auto D_s_a = new Distribution!(State, Action)(model.S().cartesian_product(model.A()), 0.0);

        auto transitions = model.T().flatten();
  
        foreach (s; model.S()) {
            foreach (a; model.A()) {
                foreach (a_p; model.A()) {
                    D_s_a[s[0], a[0]] += empirical_D_s_a_t[timestep][s[0], a_p[0]];              
                }
                D_s_a[s[0], a[0]] *= policy[timestep][s][a];
            }
        }

        return D_s_a;
    }
    
}



// TODO: Implement expectation finders; Exact, Gibbs, Matropolis-Hastings, Hybrid MCMC
// TODO: Implement LME IRL as containing one IRL problem and an expectation solver
// TODO: Implement Robost IRL as containing a Shervin style problem and an expectation solver


class LME_IRL (T ...) {

    protected Sequence_MaxEnt_Problem!(T) M;
    protected Sequence_Distribution_Computer!(T) E;
    protected double tolerance;
    protected double max_iter;
    protected bool debug_output;
    
    public this(Sequence_MaxEnt_Problem!(T) M, Sequence_Distribution_Computer!(T) E, double EM_tolerance, size_t EM_max_iter, bool debug_output = false) {
        this.M = M;
        this.E = E;
        this.tolerance = EM_tolerance;
        this.max_iter = EM_max_iter;
        this.debug_output = debug_output;
    }


    public double [] solve(Sequence!(T)[] trajectories, double [] initial_weights) {

        double [] weights = initial_weights.dup;

        size_t iters = 0;
        bool should_continue = true;
        
        do {

            auto expectation_traj = E.to_traj_distr(trajectories, weights);

            auto new_weights = M.solve(expectation_traj);


            // decide if we've converged

            should_continue = euclideanDistance(new_weights, weights, tolerance) >= tolerance;
            iters ++;
            
            weights = new_weights;

            if (debug_output) {
                import std.stdio;
                writeln("Iteration ", iters, " Weights ", new_weights);
            }
        } while (iters < max_iter && should_continue);

        return weights;
    }

}
