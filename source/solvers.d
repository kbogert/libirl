module solvers;


import std.math;
import utility;
import std.array;
import std.algorithm.comparison;
import std.algorithm;

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

    while (iterations < max_iter && err_diff > err) {

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
        
//        writeln(t, ": ", z_t, " => ", expert_features[t], " w: ", weights, " actual_w: ", actual_weights);
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
