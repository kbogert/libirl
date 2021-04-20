module solvers;


import std.math;
import utility;
import std.array;
import std.algorithm.comparison;
import std.algorithm;
import discretefunctions;
import std.typecons;
import std.random;


/*

Need:

LBFGS

Exponentiated gradient

Nelder-Mead

Primal-Dual Convex optimization method

Matrix inversion / standard matrix ops

What serves me best long term?
    Should I just import existing solvers in CPP and C? (CppNumericalSolvers project)
    Should I write my own?
    Should I port CppNumericalSolvers to D with Mir?

EXECUTIVE DECISION:

    I don't have time to do this correctly, just go with a D impl of LBFGS right now and
    implement exponentiated gradient myself, use old style nelder-mead impl.

    Section off this stuff into this file so that we don't taint the rest of the code.

    
*/


double [] exponentiatedGradientDescent(double [] expert_features, double [] initial_weights, double learning_rate, double err, size_t max_iter, size_t feature_scale, double [] delegate (double []) ff) {
        import std.stdio;

    // prep by normalizing all inputs
    auto ef_normed = expert_features.dup;
    ef_normed[] /= feature_scale;

    auto weights = initial_weights.dup;
    foreach (ref w ; weights)
        w = abs(w);

    double norm = l1norm(weights);
    if (norm != 0)
        weights[] /= norm;

    size_t iters = 0;
    double diff;
    
    do {
        
//        writeln(weights); 
        
        double [] f = ff(weights);
        f[] /= feature_scale;
//writeln(" f ", f, "\nef ", ef_normed);
        f[] -= ef_normed[];

        
        auto new_w = weights.dup;

        foreach (i; 0 .. new_w.length) {
            new_w[i] *= exp(-2 * learning_rate * f[i]);
        }
        norm = l1norm(new_w);
        if (norm != 0)
            new_w[] /= norm;


        auto temp = weights.dup;
        temp[] -= new_w[];
        diff = l2norm(temp);        

        iters ++;
        weights = new_w;
        learning_rate /= 1.05;       

//writeln(diff, " ", err);

    } while(diff > err && iters < max_iter);

    
    return weights;
}

double [] unconstrainedAdaptiveExponentiatedStochasticGradientDescent(double [][] expert_features, double nu, double err, size_t max_iter, double [] delegate (double [], size_t) ff, bool usePathLengthBounds = true, size_t moving_average_length = 5) {
//    import std.stdio;

    double [] beta = new double[expert_features[0].length * 2];
    beta[0..(beta.length / 2)] = - log(beta.length / 2 );
    beta[beta.length/2 .. $] = - log(beta.length );   
    

    double [] z_prev = new double [beta.length / 2];
    z_prev[] = 0;
    double [] w_prev = new double [beta.length / 2];
    w_prev[] = 0;

    size_t t = 0;
    size_t iterations = 0;
    double[][] moving_average_data;
    size_t moving_average_counter = 0;
    double [] err_moving_averages = new double[moving_average_length];
    foreach (ref e ; err_moving_averages) {
       e = double.max;
    }
    double err_diff = double.infinity;

    while (iterations < max_iter && (err_diff > err || iterations < moving_average_length)) {

        double [] m_t = z_prev.dup;

        if (! usePathLengthBounds && iterations > 0)
            m_t[] /= iterations;

        double [] weights = new double[beta.length];
        foreach (i ; 0 .. (beta.length / 2)) {
            weights[i] = exp(beta[i] - nu*m_t[i]);
            weights[i + (beta.length / 2)] = exp(beta[i + (beta.length / 2)] + nu*m_t[i]);
        }

        // allow for negative weights by interpreting the second half
        // of the weight vector as negative values
        double [] actual_weights = new double[beta.length / 2];
        foreach(i; 0 .. actual_weights.length) {
            actual_weights[i] = weights[i] - weights[i + actual_weights.length];
        }

        double [] z_t = ff(actual_weights, t);
//import std.stdio;        
//writeln(t, ": ", z_t, " => ", expert_features[t], " w: ", weights, " actual_w: ", actual_weights);

        z_t[] -= expert_features[t][];
            
        if (usePathLengthBounds) {
            z_prev = z_t;
        } else {
            z_prev[] += z_t[];
        }


        foreach(i; 0..(beta.length / 2)) {
            beta[i] = beta[i] - nu*z_t[i] - nu*nu*(z_t[i] - m_t[i])*(z_t[i] - m_t[i]);
            beta[i + (beta.length / 2)] = beta[i + (beta.length / 2)] + nu*z_t[i] + nu*nu*(z_t[i] - m_t[i])*(z_t[i] - m_t[i]);
        }	


        t ++;
        t %= expert_features.length;
        iterations ++;
        if (t == 0) {
            nu /= 1.04;
            err_moving_averages[moving_average_counter] = abs_average(moving_average_data);
            moving_average_counter ++;
            moving_average_counter %= moving_average_length;
            moving_average_data.length = 0;
            err_diff = stddev(err_moving_averages);
//            writeln(err_moving_averages);
//            writeln(err_diff);
//            writeln(abs_diff_average(err_moving_averages));
        }
        moving_average_data ~= z_t.dup;
        w_prev = actual_weights;   
    }
        
    return w_prev;
}




double [] unconstrainedAdaptiveExponentiatedGradientDescent(double [] expert_features, double nu, double err, size_t max_iter, double [] delegate (double []) ff, bool usePathLengthBounds = true, size_t moving_average_length = 5) {
//    import std.stdio;

    double [] beta = new double[expert_features.length * 2];
    beta[0..(beta.length / 2)] = - log(beta.length / 2 );
    beta[beta.length/2 .. $] = - log(beta.length );   
    

    double [] z_prev = new double [beta.length / 2];
    z_prev[] = 0;
    double [] w_prev = new double [beta.length / 2];
    w_prev[] = 0;

    size_t t = 0;
    size_t iterations = 0;
    double[][] moving_average_data;
    size_t moving_average_counter = 0;
    double [] err_moving_averages = new double[moving_average_length];
    foreach (ref e ; err_moving_averages) {
       e = double.max;
    }
    double err_diff = double.infinity;

    while (iterations < max_iter && (err_diff > err || iterations < moving_average_length)) {

        double [] m_t = z_prev.dup;

        if (! usePathLengthBounds && iterations > 0)
            m_t[] /= iterations;

        double [] weights = new double[beta.length];
        foreach (i ; 0 .. (beta.length / 2)) {
            weights[i] = exp(beta[i] - nu*m_t[i]);
            weights[i + (beta.length / 2)] = exp(beta[i + (beta.length / 2)] + nu*m_t[i]);
        }

        // allow for negative weights by interpreting the second half
        // of the weight vector as negative values
        double [] actual_weights = new double[beta.length / 2];
        foreach(i; 0 .. actual_weights.length) {
            actual_weights[i] = weights[i] - weights[i + actual_weights.length];
        }

        double [] z_t = ff(actual_weights);
//import std.stdio;        
//writeln(iterations, ": ", nu, ", ", z_t, " => ", expert_features, " w: ", weights, " actual_w: ", actual_weights);

        z_t[] -= expert_features[];
            
        if (usePathLengthBounds) {
            z_prev = z_t;
        } else {
            z_prev[] += z_t[];
        }


        foreach(i; 0..(beta.length / 2)) {
            beta[i] = beta[i] - nu*z_t[i] - nu*nu*(z_t[i] - m_t[i])*(z_t[i] - m_t[i]);
            beta[i + (beta.length / 2)] = beta[i + (beta.length / 2)] + nu*z_t[i] + nu*nu*(z_t[i] - m_t[i])*(z_t[i] - m_t[i]);
        }	


//        t ++;
 //       t %= expert_features.length;
        iterations ++;
//        if (t == 0) {
            nu /= 1.01;
            err_moving_averages[moving_average_counter] = l1norm(z_t);
            moving_average_counter ++;
            moving_average_counter %= moving_average_length;
            err_diff = stddev(err_moving_averages);
//            writeln(err_moving_averages, " ", err_diff);
//            writeln(err_diff);
//            writeln(abs_diff_average(err_moving_averages));
//        }
        w_prev = actual_weights;   
    }
        
    return w_prev;
}



Sequence!(Distribution!(T)) SequenceMarkovChainSmoother(T)(Sequence!(Distribution!(T)) observations, ConditionalDistribution!(T, T) transitions, Distribution!(T) initial_state) {

    
    Sequence!(Distribution!(T)) forward = new Sequence!(Distribution!(T))(observations.length);

    // forward step

    foreach(t, o_t; observations) {

        Distribution!(T) prior;
        
        if (t == 0) {
            forward[t] = tuple(new Distribution!(T)(initial_state * o_t[0]));        
        } else {
            forward[t] = tuple(new Distribution!(T)(sumout((transitions * forward[t-1][0]).reverse_params()) * o_t[0]));        
        }
    }
    // backward step

    Sequence!(Distribution!(T)) backward = new Sequence!(Distribution!(T))(observations.length);

    backward[$] = tuple(new Distribution!(T)(observations[$][0].param_set(), 1.0));
    foreach_reverse(t, o_t; observations) {

        if (t > 0) {
            backward[t-1] = tuple(new Distribution!(T)((sumout(((transitions.flatten() * o_t[0]) * backward[t][0]))) ));
        }
    }

    Sequence!(Distribution!(T)) returnval = new Sequence!(Distribution!(T))(observations.length);

    foreach(t, f_t ; forward) {
        returnval[t] = tuple(new Distribution!(T)(f_t[0] * backward[t][0]));
        returnval[t][0].normalize();
    }
    
    return returnval;

    
}


Sequence!(Distribution!(T)) MarkovGibbsSampler(T)(Sequence!(Distribution!(T)) observations, ConditionalDistribution!(T, T) transitions, Distribution!(T) initial_state, size_t burn_in_samples, size_t total_samples, const bool delegate(Sequence!(Distribution!(T)) , size_t) convergence_check = null) {

    double [Tuple!T][] returnval_arr = new double[Tuple!T][observations.length];

    Sequence!(Distribution!(T)) returnval = new Sequence!(Distribution!(T))(observations.length);
    foreach(t ; 0 .. observations.length) {
        returnval[t] = tuple(new Distribution!(T)(observations[t][0].param_set(), 0.0));

    }    
    
    Sequence!(T) currentState = new Sequence!(T)(observations.length);

    // create initial state

    bool allSampled = false;
    size_t attempts = 0;
    do {
        foreach(t; 0 .. observations.length) {

            try {
                if (t == 0) {
                    currentState[t] = new Distribution!(T)((initial_state * observations[t][0])).sample();
            
                } else {
                    currentState[t] = new Distribution!(T)((transitions[currentState[t-1]] * observations[t][0])).sample();

                }
            } catch (Exception e) {
                // assume this is due to distributions with all zeros
                allSampled = false;
                attempts ++;
                if (attempts > observations.length * initial_state.param_set().size() * 1000) {
                    import std.conv;
                    throw new Exception("Unable to create initial trajectory sample after" ~ to!string(observations.length * initial_state.param_set().size() * 1000) ~ " attempts");
                }
                break;
            }
            allSampled = true;
        }
    } while (!allSampled);    

    foreach(i; 0 .. (burn_in_samples + total_samples)) {

        auto position = i % observations.length;

        Function!(Tuple!T, T) chooser;
        if (position != observations.length - 1) {

            Tuple!(T) [Tuple!(T)] chooser_arr;
            
            foreach( t; observations[position][0].param_set()) {
                chooser_arr[t] = currentState[position + 1];
            }
            
            chooser = new Function!(Tuple!T, T)(observations[position][0].param_set(), chooser_arr);
        }


        
        Tuple!T newSample;
        if (position == 0) {
            newSample = new Distribution!(T)(((transitions * (initial_state * observations[position][0])).reverse_params()).apply(chooser)).sample();
        } else if (position == observations.length - 1) {
            newSample = new Distribution!(T)(sumout((transitions * (transitions[currentState[position-1]] * observations[position][0])).reverse_params())).sample();
        } else {
            newSample = new Distribution!(T)(((transitions * (transitions[currentState[position-1]] * observations[position][0])).reverse_params()).apply(chooser)).sample();
        }

        if (i > burn_in_samples) {

            returnval[position][0][newSample] += 1;
            
            if (convergence_check ! is null && 
                convergence_check(returnval, i - burn_in_samples))
                break;
        }
        currentState[position] = newSample;
    }

    foreach(entry; returnval) {
        entry[0].normalize();
    }    


    return returnval;
}


Sequence!(Distribution!(T)) HybridMCMC(T)(Sequence!(Distribution!(T)) observations, ConditionalDistribution!(T, T) transitions, Distribution!(T) initial_state, Sequence!(ConditionalDistribution!(T, T)) proposal_distributions, size_t burn_in_samples, size_t total_samples, const bool delegate(Sequence!(Distribution!(T)) , size_t) convergence_check = null) {


    double [Tuple!T][] returnval_arr = new double[Tuple!T][observations.length];

    Sequence!(Distribution!(T)) returnval = new Sequence!(Distribution!(T))(observations.length);
    foreach(t ; 0 .. observations.length) {
        returnval[t] = tuple(new Distribution!(T)(observations[t][0].param_set(), 0.0));

        // validate proposal distribution

        foreach (T1; observations[t][0].param_set()) {
            if (proposal_distributions[t][0][T1].entropy() == 0.0) {
                throw new Exception("Proposal distribution cannot have zero entropy, at least two tokens must have non-zero probability.");
            }
        }            
    }    
    
    Sequence!(T) currentState = new Sequence!(T)(observations.length);

    // create initial state

    bool allSampled = false;
    size_t attempts = 0;
    do {
        foreach(t; 0 .. observations.length) {

            try {
                if (t == 0) {
                    currentState[t] = new Distribution!(T)((initial_state * observations[t][0])).sample();
            
                } else {
                    currentState[t] = new Distribution!(T)((transitions[currentState[t-1]] * observations[t][0])).sample();

                }
            } catch (Exception e) {
                // assume this is due to distributions with all zeros
                allSampled = false;
                attempts ++;
                if (attempts > observations.length * initial_state.param_set().size() * 1000) {
                    import std.conv;
                    throw new Exception("Unable to create initial trajectory sample after" ~ to!string(observations.length * initial_state.param_set().size() * 1000) ~ " attempts");
                }
                break;
            }
            allSampled = true;
        }
    } while (!allSampled);
    
    foreach(i; 0 .. (burn_in_samples + total_samples)) {

        auto position = i % observations.length;

        Tuple!T newSample;
        do {
            newSample = proposal_distributions[position][0][currentState[position]].sample();
        } while (newSample == currentState[position]);
       
        double newSampleProb;
        double oldSampleProb;
        
        if (position == 0) {
            newSampleProb = (initial_state[newSample] * observations[position][0][newSample] * transitions[newSample][currentState[position+1]]) * proposal_distributions[position][0][newSample][currentState[position]];
            oldSampleProb = (initial_state[currentState[position]] * observations[position][0][currentState[position]] * transitions[currentState[position]][currentState[position+1]]) * proposal_distributions[position][0][currentState[position]][newSample];
        } else if (position == observations.length - 1) {
            newSampleProb = ( observations[position][0][newSample] * transitions[currentState[position-1]][newSample]) * proposal_distributions[position][0][newSample][currentState[position]];
            oldSampleProb = ( observations[position][0][currentState[position]] * transitions[currentState[position-1]][currentState[position]]) * proposal_distributions[position][0][currentState[position]][newSample];
        } else {
            newSampleProb = ( transitions[currentState[position-1]][newSample] * observations[position][0][newSample] * transitions[newSample][currentState[position+1]]) * proposal_distributions[position][0][newSample][currentState[position]];
            oldSampleProb = ( transitions[currentState[position-1]][currentState[position]] * observations[position][0][currentState[position]] * transitions[currentState[position]][currentState[position+1]]) * proposal_distributions[position][0][currentState[position]][newSample];
        }
        
        double acc = (fmin(1, newSampleProb / oldSampleProb ));

        if (uniform01() <= acc) {
            currentState[position] = newSample;
        } 

        if (i > burn_in_samples) {

            returnval[position][0][currentState[position]] += 1;

            if (convergence_check ! is null && 
                    convergence_check(returnval, i - burn_in_samples))
                break;
        }

    }

    foreach(entry; returnval) {
        entry[0].normalize();
    }    


    return returnval;
}


