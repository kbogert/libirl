module gridworldirltest;

import gridworld;
import discretemdp;
import discretefunctions;
import trajectories;
import tested;
import std.random;
import std.math;
import std.conv;
import std.algorithm;
import std.stdio;
import std.array;
import std.typecons;

import maxentIRL;
import utility;
import analysis;


@name("MaxEntIRL exact random reward function recovery test")
unittest {

    
    int sizeX = 5;
    int sizeY = 5;
    double gamma = 0.95;
    double value_error = 0.1;
    int samples = 1000 * sizeX * sizeY;
    double tolerance = 0.0001;

/*    version(fullunittest) {
        samples = samples * 100;
        tolerance *= 0.01;
    }*/

    Tuple!(int,int) [] terminals;

    terminals ~= tuple(0, 0);
    terminals ~= tuple(sizeX - 1, sizeY - 1);
    
    GridWorldStateSpaceWithTerminal states = new GridWorldStateSpaceWithTerminal(sizeX, sizeY, terminals);
    GridWorldActionSpace actions = new GridWorldActionSpace();

    // features only defined on terminal states

    Function!(double [], State, Action) features = new Function!(double [], State, Action)(states.cartesian_product(actions), new double[terminals.length]);
    auto i = 0;
    foreach (s ; states) {
        double [] f = new double[terminals.length];
        f[] = 0;
        foreach(term ; terminals) {
            if ((cast(GridWorldState)s[0]).getX() == term[0] && (cast(GridWorldState)s[0]).getY() == term[1]) {
                f[i] = 1;
                i ++;
                break;
            }
        }
        foreach (a ; actions) {

            features[ s[0] , a[0] ] = f;
        }
    }
    
    /*
    double[] tmpArray = new double[states.size];
    tmpArray[] = 0;

    Function!(double [], State, Action) features = new Function!(double [], State, Action)(states.cartesian_product(actions), tmpArray);

    auto i = 0;
    foreach (s ; states) {
        foreach (a ; actions) {
            features[ s[0] , a[0] ] = tmpArray.dup;
            features[ s[0] , a[0] ][max(0, i - 1)] = 1.0;
            features[ s[0] , a[0] ][i] = 1.0;
            features[ s[0] , a[0] ][min(states.size() - 1, i + 1)] = 1.0;
        }
        i ++;
    }
*/


    int iterations = 2;
    version (fullunittest) {
        iterations = 20;
    }

    auto transitions = new ConditionalDistribution!(State, State, Action)(states, states.cartesian_product(actions));

    foreach (s ; states) {
        foreach (a ; actions) {

            if (s[0].isTerminal()) {
                transitions[s[0], a[0]] = new Distribution!(State)(states, 0.0);
                transitions[tuple(s[0], a[0])][s[0]] = 1.0;
                continue;
            }
            
            auto newState = (cast(GridWorldAction)a[0]).getIdealStateFor(cast(GridWorldState)s[0]);

            Distribution!State ds = new Distribution!(State)(states, 0.0);

            if (states.contains(cast(State)newState)) {
                ds[newState] = 1.0;
            } else {
                ds[s[0]] = 1.0;
            }

            ds.normalize();
            
            transitions[s[0], a[0]] = ds;
        }
    }
    
    foreach (iter ; 0 .. iterations ) {

        double [] weights;
        weights.length = terminals.length;
        foreach (ref w ; weights) {
            w = uniform(0.0, 10.0);
        }
        
        auto lr = new LinearReward(features, weights);
        auto model = new SoftMaxModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform), value_error * max ( max( lr.toFunction())),  sizeX * sizeY * 10);

        
        auto V = soft_max_value_iteration(model, value_error * max ( max( lr.toFunction())) , sizeX * sizeY * 10);
//        auto policy = soft_max_policy(V, model);
        auto policy = model.getPolicy();
//        auto V = value_iteration(model, value_error * max ( max( lr.toFunction())) , sizeX * sizeY * 10);
//        auto policy = to_stochastic_policy(optimum_policy(V, model), actions);

        Sequence!(State, Action) [] trajectories;
        foreach (j ; 0 .. samples ) {
            trajectories ~= simulate(model, policy, (sizeX + sizeY), model.initialStateDistribution() );
        }
//        writeln(trajectories);

        // Perform MaxEntIrl

        auto maxEntIRL = new MaxEntIRL_Ziebart_exact(model, lr, tolerance, weights);
        
        double [] found_weights = maxEntIRL.solve (traj_to_traj_distr(trajectories, model), iter % 2 == 0); // alternate solvers

        double err = calcInverseLearningError(model, new LinearReward(features, weights), new LinearReward(features, found_weights), tolerance, sizeX * sizeY * 10);

        // make sure the inverse error is low, like less than a state's value
        assert(err >= 0 && err < V[states.getOne()], "MaxEntIRL found bad solution (err: " ~ to!string(err) ~ ", " ~  to!string(iter) ~ ") : " ~ to!string(found_weights) ~ " correct: " ~ to!string(weights));
    }    

}


@name("MaxCausalEntIRL-Ziebart exact random reward function recovery test")
unittest {

    
    int sizeX = 5;
    int sizeY = 5;
    double gamma = 0.95;
    double value_error = 0.1;
    int samples = 1000 * sizeX * sizeY;
    double tolerance = 0.0001;

/*    version(fullunittest) {
        samples = samples * 100;
        tolerance *= 0.01;
    }*/
/*
    Tuple!(int,int) [] terminals;

    terminals ~= tuple(0, 0);
    terminals ~= tuple(sizeX - 1, sizeY - 1);
*/
    
//    GridWorldStateSpaceWithTerminal states = new GridWorldStateSpaceWithTerminal(sizeX, sizeY, terminals);
    GridWorldStateSpace states = new GridWorldStateSpace(sizeX, sizeY);
    GridWorldActionSpace actions = new GridWorldActionSpace();

    // features only defined on terminal states
/*
    Function!(double [], State, Action) features = new Function!(double [], State, Action)(states.cartesian_product(actions), new double[terminals.length]);
    auto i = 0;
    foreach (s ; states) {
        double [] f = new double[terminals.length];
        f[] = 0;
        foreach(term ; terminals) {
            if ((cast(GridWorldState)s[0]).getX() == term[0] && (cast(GridWorldState)s[0]).getY() == term[1]) {
                f[i] = 1;
                i ++;
                break;
            }
        }
        foreach (a ; actions) {

            features[ s[0] , a[0] ] = f;
        }
    }
*/    
    
    double[] tmpArray = new double[states.size];
    tmpArray[] = 0;

    Function!(double [], State, Action) features = new Function!(double [], State, Action)(states.cartesian_product(actions), tmpArray);

    auto i = 0;
    foreach (s ; states) {
        foreach (a ; actions) {
            features[ s[0] , a[0] ] = tmpArray.dup;
            features[ s[0] , a[0] ][max(0, i - 1)] = 1.0;
            features[ s[0] , a[0] ][i] = 1.0;
            features[ s[0] , a[0] ][min(states.size() - 1, i + 1)] = 1.0;
        }
        i ++;
    }

    int iterations = 2;
    version (fullunittest) {
        iterations = 20;
    }
    
    foreach (iter ; 0 .. iterations ) {

        double [] weights;
        weights.length = features[features.param_set.getOne()].length;
        foreach (ref w ; weights) {
            w = uniform(0.0, 10.0);
        }

        // randomly generate transition function
        ConditionalDistribution!(State, State, Action) transitions;
        
//        transitions = build_simple_transition_function(states, actions, 1.0, & otherActionsErrorFunction);

//        transitions = build_simple_transition_function(states, actions, uniform(.4, 1.0), & otherActionsErrorFunction);

        // more awesomely generate random transition function        
        transitions = new ConditionalDistribution!(State, State, Action)(states, states.cartesian_product(actions));

        foreach (s ; states) {
            foreach (a ; actions) {
                transitions[s[0], a[0]] = new Distribution!(State)(states, DistInitType.Exponential, 15.0);
            }
        }

//        writeln(transitions);
                
        auto lr = new LinearReward(features, weights);
        auto model = new SoftMaxModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform),  value_error * max ( max( lr.toFunction())) , sizeX * sizeY * 10);

        
        auto V = soft_max_value_iteration(model, value_error * max ( max( lr.toFunction())) , sizeX * sizeY * 10);
        auto policy = soft_max_policy(V, model);
//        auto V = value_iteration(model, value_error * max ( max( lr.toFunction())) , sizeX * sizeY * 10);
//        auto policy = to_stochastic_policy(optimum_policy(V, model), actions);

        Sequence!(State, Action) [] trajectories;
        foreach (j ; 0 .. samples ) {
            trajectories ~= simulate(model, policy, (sizeX + sizeY), model.initialStateDistribution() );
        }
//        writeln(trajectories);

        // Perform MaxCausalEntIrl

        auto maxCausalEntIRL = new MaxCausalEntIRL_Ziebart(model, lr, tolerance, weights);
        
        double [] found_weights = maxCausalEntIRL.solve (traj_to_traj_distr(trajectories, model), iter % 2 == 0); // alternate solvers


        double err = calcInverseLearningError(model, new LinearReward(features, weights), new LinearReward(features, found_weights), tolerance, sizeX * sizeY * 10);

        // make sure the inverse error is low, like less than a state's value
        assert(err >= 0 && err < V[states.getOne()], "MaxCausalEntIRL found bad solution (err: " ~ to!string(err) ~ ", " ~  to!string(iter) ~ ") : " ~ to!string(found_weights) ~ " correct: " ~ to!string(weights));
    }    

}


@name("MaxCausalEntIRL_Inf and approximations random reward function recovery test")
unittest {

    
    int sizeX = 5;
    int sizeY = 5;
    double gamma = 0.95;
    double value_error = 0.1;
    int samples = 1000 * sizeX * sizeY;
    double tolerance = 0.0001;

/*    version(fullunittest) {
        samples = samples * 100;
        tolerance *= 0.01;
    }*/
/*
    Tuple!(int,int) [] terminals;

    terminals ~= tuple(0, 0);
    terminals ~= tuple(sizeX - 1, sizeY - 1);
*/
    
//    GridWorldStateSpaceWithTerminal states = new GridWorldStateSpaceWithTerminal(sizeX, sizeY, terminals);
    GridWorldStateSpace states = new GridWorldStateSpace(sizeX, sizeY);
    GridWorldActionSpace actions = new GridWorldActionSpace();

    // features only defined on terminal states
/*
    Function!(double [], State, Action) features = new Function!(double [], State, Action)(states.cartesian_product(actions), new double[terminals.length]);
    auto i = 0;
    foreach (s ; states) {
        double [] f = new double[terminals.length];
        f[] = 0;
        foreach(term ; terminals) {
            if ((cast(GridWorldState)s[0]).getX() == term[0] && (cast(GridWorldState)s[0]).getY() == term[1]) {
                f[i] = 1;
                i ++;
                break;
            }
        }
        foreach (a ; actions) {

            features[ s[0] , a[0] ] = f;
        }
    }
*/    
    
    double[] tmpArray = new double[states.size];
    tmpArray[] = 0;

    Function!(double [], State, Action) features = new Function!(double [], State, Action)(states.cartesian_product(actions), tmpArray);

    auto i = 0;
    foreach (s ; states) {
        foreach (a ; actions) {
            features[ s[0] , a[0] ] = tmpArray.dup;
            features[ s[0] , a[0] ][max(0, i - 1)] = 1.0;
            features[ s[0] , a[0] ][i] = 1.0;
            features[ s[0] , a[0] ][min(states.size() - 1, i + 1)] = 1.0;
        }
        i ++;
    }

    int iterations = 2;
    version (fullunittest) {
        iterations = 20;
    }
    
    foreach (iter ; 0 .. iterations ) {

        double [] weights;
        weights.length = features[features.param_set.getOne()].length;
        foreach (ref w ; weights) {
            w = uniform(0.0, 10.0);
        }

        // randomly generate transition function
        ConditionalDistribution!(State, State, Action) transitions;
        
//        transitions = build_simple_transition_function(states, actions, 1.0, & otherActionsErrorFunction);

//        transitions = build_simple_transition_function(states, actions, uniform(.4, 1.0), & otherActionsErrorFunction);

        // more awesomely generate random transition function        
        transitions = new ConditionalDistribution!(State, State, Action)(states, states.cartesian_product(actions));

        foreach (s ; states) {
            foreach (a ; actions) {
                transitions[s[0], a[0]] = new Distribution!(State)(states, DistInitType.Exponential, 15.0);
            }
        }

//        writeln(transitions);
                
        auto lr = new LinearReward(features, weights);
        auto model = new SoftMaxModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform),  value_error * max ( max( lr.toFunction())) , sizeX * sizeY * 10);

        
        auto V = soft_max_value_iteration(model, value_error * max ( max( lr.toFunction())) , sizeX * sizeY * 10);
        auto policy = soft_max_policy(V, model);
//        auto V = value_iteration(model, value_error * max ( max( lr.toFunction())) , sizeX * sizeY * 10);
//        auto policy = to_stochastic_policy(optimum_policy(V, model), actions);

        Sequence!(State, Action) [] trajectories;
        foreach (j ; 0 .. samples ) {
            trajectories ~= simulate(model, policy, (sizeX + sizeY), model.initialStateDistribution() );
        }
//        writeln(trajectories);

        // Perform MaxCausalEntIrl_Inf

        auto maxCausalEntIRL = new MaxCausalEntIRL_InfMDP(model, lr, tolerance, weights);
        
        double [] found_weights = maxCausalEntIRL.solve (traj_to_traj_distr(trajectories, model), iter % 2 == 0); // alternate solvers


        double err = calcInverseLearningError(model, new LinearReward(features, weights), new LinearReward(features, found_weights), tolerance, sizeX * sizeY * 10);

        // make sure the inverse error is low, like less than a state's value
        assert(err >= 0 && err < V[states.getOne()], "MaxCausalEntIRL_Inf found bad solution (err: " ~ to!string(err) ~ ", " ~  to!string(iter) ~ ") : " ~ to!string(found_weights) ~ " correct: " ~ to!string(weights));


        // Perform MaxCausalEntIrl_Approx
        
        maxCausalEntIRL = new MaxCausalEntIRL_SGDApprox(model, lr, tolerance, weights);
        
        found_weights = maxCausalEntIRL.solve (traj_to_traj_distr(trajectories, model), iter % 2 == 0); // alternate solvers


        err = calcInverseLearningError(model, new LinearReward(features, weights), new LinearReward(features, found_weights), tolerance, sizeX * sizeY * 10);

        // make sure the inverse error is low, like less than a state's value
        assert(err >= 0 && err < V[states.getOne()], "MaxCausalEntIRL_Approx found bad solution (err: " ~ to!string(err) ~ ", " ~  to!string(iter) ~ ") : " ~ to!string(found_weights) ~ " correct: " ~ to!string(weights));


        // Perform MaxCausalEntIrl_Empirical
        
        maxCausalEntIRL = new MaxCausalEntIRL_SGDEmpirical(model, lr, tolerance, weights);
        
        found_weights = maxCausalEntIRL.solve (traj_to_traj_distr(trajectories, model), iter % 2 == 0); // alternate solvers


        err = calcInverseLearningError(model, new LinearReward(features, weights), new LinearReward(features, found_weights), tolerance, sizeX * sizeY * 10);

        // make sure the inverse error is low, like less than a state's value
        assert(err >= 0 && err < V[states.getOne()], "MaxCausalEntIRL_Empirical found bad solution (err: " ~ to!string(err) ~ ", " ~  to!string(iter) ~ ") : " ~ to!string(found_weights) ~ " correct: " ~ to!string(weights));
    }    

}
