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

double [] unconstrainedAdaptiveExponentiatedStochasticGradientDescent(double [][] expert_features, double nu, double err, size_t max_iter, double [] delegate (double [], size_t) ff, bool usePathLengthBounds = true, size_t moving_average_length = 5, bool debugOn = false, double [] initial_params = null, double weight_limit = 100, void delegate (double) nu_out = null) {
//    import std.stdio;

    double [] beta = new double[expert_features[0].length * 2];
    if (! (initial_params is null)) {
//        writeln(initial_params);
        beta[] = 0;
        foreach(i, ip; initial_params) {
            if (ip > 0) {
                beta[i] = log(1.5*ip);
                beta[i + beta.length/2] = log(0.5*ip);
            } else{
                beta[i] = log(0.5*-ip);                
                beta[i + beta.length/2] = log(1.5*-ip); 
            }
        }
//        writeln(beta);
    } else {
        beta[0..(beta.length / 2)] = - log(beta.length / 2 );
        beta[beta.length/2 .. $] = - log(beta.length );   
    }
    

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
    double [][] oscillation_check_data;
    foreach(i; 0 .. moving_average_length) {
        oscillation_check_data ~= new double[beta.length /2];
        oscillation_check_data[i][] = 0;
    }

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
        if (debugOn) {
            import std.stdio;        
            writeln(t, ": ", z_t, " vs ", expert_features[t], " weights: ", actual_weights);
        }
        
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
            nu /= 1.0005;
            err_moving_averages[moving_average_counter] = abs_average(moving_average_data);
            moving_average_data.length = 0;
            err_diff = stddev(err_moving_averages);
            if (detect_oscillation(oscillation_check_data, moving_average_counter, actual_weights)) {
                nu *= 0.9;
                if (debugOn) {
                    import std.stdio;
                    writeln("correct nu down");
                }
            }
            moving_average_counter ++;
            moving_average_counter %= moving_average_length;
            if (debugOn) {
                import std.stdio;
                writeln("SGD std dev ", err_diff, " vs ", err, ", iterations: ", iterations, " of ", max_iter);
//               writeln(abs_diff_average(err_moving_averages));
            }
        }
        moving_average_data ~= z_t.dup;
        w_prev = actual_weights;  

        if (l2norm(w_prev) > weight_limit) {
            // weight magnitude too large, probably can't solve the problem
            break;
        } 
    }

    if (nu_out != null) {
        nu_out(nu);
    }
        
    return w_prev;
}




double [] unconstrainedAdaptiveExponentiatedGradientDescent(double [] expert_features, double nu, double err, size_t max_iter, double [] delegate (double []) ff, bool usePathLengthBounds = true, size_t moving_average_length = 5, bool debugOn = false, double [] initial_params = null, double weight_limit = 100, void delegate (double) nu_out = null) {
    import std.stdio;

    double [] beta = new double[expert_features.length * 2];
    double [] z_prev = new double [beta.length / 2];
    z_prev[] = 0;
    double [] w_prev = new double [beta.length / 2];
    w_prev[] = 0;

    if (! (initial_params is null)) {
//        writeln(initial_params);
        beta[] = 0;
        foreach(i, ip; initial_params) {
            if (ip > 0) {
                beta[i] = log(1.5*ip);
                beta[i + beta.length/2] = log(0.5*ip);
            } else{
                beta[i] = log(0.5*-ip);                
                beta[i + beta.length/2] = log(1.5*-ip); 
            }
        }
//        writeln(beta);
        z_prev = ff(initial_params);
        z_prev[] -= expert_features[];

    } else {
        beta[0..(beta.length / 2)] = - log(beta.length / 2 );
        beta[beta.length/2 .. $] = - log(beta.length );
        
        double [] weights = new double[beta.length];
        foreach (i ; 0 .. (beta.length / 2)) {
            weights[i] = exp(beta[i]);
            weights[i + (beta.length / 2)] = exp(beta[i + (beta.length / 2)]);
        }
        z_prev = ff(weights);
        z_prev[] -= expert_features[];
    }

    size_t t = 0;
    size_t iterations = 0;
    size_t moving_average_counter = 0;
    double [] err_moving_averages = new double[moving_average_length];
    foreach (ref e ; err_moving_averages) {
       e = double.max;
    }
    double err_diff = double.infinity;
    double [][] oscillation_check_data;
    foreach(i; 0 .. moving_average_length) {
        oscillation_check_data ~= new double[beta.length /2];
        oscillation_check_data[i][] = 0;
    }
    
    while (iterations < max_iter && (err_diff > err || iterations < moving_average_length * 2)) {

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

        if (debugOn) {
            import std.stdio;        
            writeln(nu, " ", iterations, ": ", z_t, " vs ", expert_features, " weights: ", actual_weights);
        }
        
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
            //nu /= 1.0001;
            nu += 0.00001;
            err_moving_averages[moving_average_counter] = vect_sensitive_l1norm(z_t);
            err_diff = stddev(err_moving_averages);           

            if (detect_oscillation(oscillation_check_data, moving_average_counter, actual_weights)) {
                nu *= 0.99;
                if (debugOn)
                    writeln("correct nu down");
            } /*else if ((old_err_diff - err_diff) / err_diff > 0.4) {
                nu /= 0.995;
                writeln("correct nu up");
            }*/
            moving_average_counter ++;
            moving_average_counter %= moving_average_length;
            if (debugOn) {
                import std.stdio;
//                writeln(err_moving_averages);
                writeln(beta, " GD std dev ", err_diff, " vs ", err, ", iterations: ", iterations, " of ", max_iter);
//               writeln(abs_diff_average(err_moving_averages));
            }
//            writeln(err_moving_averages, " ", err_diff);
//            writeln(err_diff);
//            writeln(abs_diff_average(err_moving_averages));
//        }
        w_prev = actual_weights;   

        if (l2norm(w_prev) > weight_limit) {
            // weight magnitude too large, probably can't solve the problem
            break;
        }
    }
     if (debugOn) {
                import std.stdio;
//                writeln("GD std dev ", err_diff, " vs ", err, ", iterations: ", iterations, " of ", max_iter);
            writeln(iterations, ": ", z_prev, " vs ", expert_features, " weights: ", w_prev);
//               writeln(abs_diff_average(err_moving_averages));
            }   
    if (nu_out != null) {
        nu_out(nu);
    }
        
    return w_prev;
}

bool detect_oscillation(ref double [][] oscillation_data, size_t array_ptr, double [] new_weights) {
    oscillation_data[array_ptr] = new_weights;

    foreach( o; 0 .. new_weights.length) {
        bool [] arrows = new bool[oscillation_data.length];
        arrows[] = true;
        foreach(osc; array_ptr + 1 .. 1 + array_ptr + oscillation_data.length) {
            if (oscillation_data[osc % oscillation_data.length][o] > oscillation_data[(osc + 1) % oscillation_data.length][o])
                arrows[osc % oscillation_data.length] = false;
        }
        int changeCount = 0;
        foreach(osc; array_ptr + 1 .. 1 + array_ptr + oscillation_data.length) {
            if (arrows[osc % oscillation_data.length] != arrows[(osc + 1) % oscillation_data.length])
                changeCount ++;
        }

        if (changeCount >= 3) {
            return true;
        }
                
    }
    return false;
            
}

double [] nonNegativeUnconstrainedAdaptiveExponentiatedGradientDescent(double [] expert_features, double nu, double err, size_t max_iter, double [] delegate (double []) ff, bool usePathLengthBounds = true, size_t moving_average_length = 5, bool debugOn = false) {
//    import std.stdio;

    double [] beta = new double[expert_features.length];
    beta[0..(beta.length)] = - log(beta.length);  
    

    double [] z_prev = new double [beta.length];
    z_prev[] = 0;
    double [] w_prev = new double [beta.length];
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
        foreach (i ; 0 .. beta.length) {
            weights[i] = exp(beta[i] - nu*m_t[i]);
        }

        double [] z_t = ff(weights);

        
        z_t[] -= expert_features[];
        foreach(ref z; z_t) {
//            z = min(0, z);
            if (z < 0) {
                z = -(1.0 / 10) * log(-(z - 1));
            }

        }
        if (debugOn) {
            import std.stdio;        
            writeln(iterations, ": ", z_t, " vs ", expert_features, " weights: ", weights);
        }
            
        if (usePathLengthBounds) {
            z_prev = z_t;
        } else {
            z_prev[] += z_t[];
        }

        foreach(i; 0..beta.length) {
            beta[i] = beta[i] - nu*z_t[i] - nu*nu*(z_t[i] - m_t[i])*(z_t[i] - m_t[i]);
            if (isNaN(beta[i])) {
                beta[i] = 0;
            }
        }	


//        t ++;
 //       t %= expert_features.length;
        iterations ++;
//        if (t == 0) {
            nu /= 1.005;
            err_moving_averages[moving_average_counter] = l1norm(z_t);
            moving_average_counter ++;
            moving_average_counter %= moving_average_length;
            err_diff = stddev(err_moving_averages);
/*            if (debugOn) {
                import std.stdio;
                writeln("GD std dev ", err_diff, " vs ", err, ", iterations: ", iterations, " of ", max_iter);
//               writeln(abs_diff_average(err_moving_averages));
            }*/
//            writeln(err_moving_averages, " ", err_diff);
//            writeln(err_diff);
//            writeln(abs_diff_average(err_moving_averages));
//        }
        w_prev = weights;
    }
            if (debugOn) {
                import std.stdio;
                writeln("GD std dev ", err_diff, " vs ", err, ", iterations: ", iterations, " of ", max_iter);
//               writeln(abs_diff_average(err_moving_averages));
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


Sequence!(Distribution!(T)) MarkovGibbsSampler(T)(Sequence!(Distribution!(T)) observations, ConditionalDistribution!(T, T) transitions, Distribution!(T) initial_state, size_t burn_in_samples, size_t total_samples, const bool delegate(Sequence!(Distribution!(T)) , size_t) convergence_check = null, bool debugOn = false) {

    double [Tuple!T][] returnval_arr = new double[Tuple!T][observations.length];

    Sequence!(Distribution!(T)) returnval = new Sequence!(Distribution!(T))(observations.length);
    foreach(t ; 0 .. observations.length) {
        returnval[t] = tuple(new Distribution!(T)(observations[t][0].param_set(), 0.0));

    }    
    
    Sequence!(T) currentState = new Sequence!(T)(observations.length);

    // create initial state
    if (debugOn) {
        import std.stdio;
        writeln("GibbsMCMC performing initial sample");
    }
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
      
    if (debugOn) {
        import std.stdio;
        writeln("GibbsMCMC initial sample complete");
    }
    
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
            if (debugOn) {
                import std.stdio;

                write("\r", "GibbsMCMC: ", (cast(double)i) / (burn_in_samples + total_samples) * 100, "%");

            }
        }
        currentState[position] = newSample;
    }

    foreach(entry; returnval) {
        entry[0].normalize();
    }    

    if (debugOn) {
        import std.stdio;
        writeln();
    }

    return returnval;
}


Sequence!(Distribution!(T)) HybridMCMC(T)(Sequence!(Distribution!(T)) observations, ConditionalDistribution!(T, T) transitions, Distribution!(T) initial_state, Sequence!(ConditionalDistribution!(T, T)) proposal_distributions, size_t burn_in_samples, size_t total_samples, const bool delegate(Sequence!(Distribution!(T)) , size_t) convergence_check = null, bool debugOn = false) {


    double [Tuple!T][] returnval_arr = new double[Tuple!T][observations.length];

    Sequence!(Distribution!(T)) returnval = new Sequence!(Distribution!(T))(observations.length);

    foreach(t ; 0 .. observations.length) {
        returnval[t] = tuple(new Distribution!(T)(observations[t][0].param_set(), 0.0));

/*        // validate proposal distribution

        foreach (T1; observations[t][0].param_set()) {
            if (proposal_distributions[t][0][T1].entropy() == 0.0) {
                throw new Exception("Proposal distribution cannot have zero entropy, at least two tokens must have non-zero probability.");
            }
        }            */
    }    
    
    Sequence!(T) currentState = new Sequence!(T)(observations.length);

    // create initial state

    if (debugOn) {
        import std.stdio;
        writeln("HybridMCMC performing initial sample");
    }
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


/*void print_sequence(Sequence!(T) seq) {
    import std.stdio;

    foreach (t, timestep; seq) {

        write("<", timestep[0][0], " ", timestep[0][1], ">, ");
        
    }
    writeln();

}                
                print_sequence(currentState);*/
                if (attempts > observations.length * initial_state.param_set().size() * 1000) {
                    import std.conv;
                    throw new Exception("Unable to create initial trajectory sample after" ~ to!string(observations.length * initial_state.param_set().size() * 1000) ~ " attempts");
                }
                break;
            }
            allSampled = true;
        }
    } while (!allSampled);

    if (debugOn) {
        import std.stdio;
        writeln("HybridMCMC initial sample complete");
    }

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

            if (debugOn) {
                import std.stdio;

                write("\r", "HybridMCMC: ", (cast(double)i) / (burn_in_samples + total_samples) * 100, "%");

            }
        }

    }

    foreach(entry; returnval) {
        entry[0].normalize();
    }    

    if (debugOn) {
        import std.stdio;
        writeln();
    }
    return returnval;
}


