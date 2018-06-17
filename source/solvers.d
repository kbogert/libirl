module solvers;


import std.math;
import utility;

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

writeln(expert_features, " ", feature_scale);
    // prep by normalizing all inputs
    auto ef_normed = expert_features.dup;
//    ef_normed[] /= feature_scale;

    auto weights = initial_weights.dup;
    foreach (ref w ; weights)
        w = abs(w);

    double norm = l1norm(weights);
    if (norm != 0)
        weights[] /= norm;

    size_t iters = 0;
    double diff;
    
    do {
        
        writeln(weights); 
        
        double [] f = ff(weights);
        f[] /= feature_scale;
writeln(" f ", f, "\nef ", ef_normed);
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

writeln(diff, " ", err);

    } while(diff > err && iters < max_iter);

    
    return weights;
}

double [] unconstrainedAdaptiveExponentiatedStochasticGradientDescent(double [][] expert_features, double learning_rate, double err, size_t max_iter, double [] delegate (double [], size_t)) {

    return expert_features[0];
}
