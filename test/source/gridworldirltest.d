module gridworldirltest;

import gridworld;
import discretemdp;
import discretefunctions;
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
    double value_error = 0.001;
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
        double [] f = minimallyInitializedArray!(double[])(terminals.length);
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
    double[] tmpArray = minimallyInitializedArray!(double[])(states.size);

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
        iterations = 50;
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
//        weights[] /= l1norm(weights);
//        writeln(weights);
        
        auto lr = new LinearReward(features, weights);
        auto model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform));

        
        auto V = soft_max_value_iteration(model, value_error * max ( max( lr.toFunction())) , sizeX * sizeY * 10);
//        auto V = value_iteration(model, value_error * max ( max( lr.toFunction())) , sizeX * sizeY * 10);
        auto policy = soft_max_policy(V, model);
//        auto policy = to_stochastic_policy(optimum_policy(V, model), actions);

        Sequence!(State, Action) [] trajectories;
        foreach (j ; 0 .. samples ) {
            trajectories ~= simulate(model, policy, (sizeX + sizeY), model.initialStateDistribution() );
        }
//        writeln(trajectories);

        // Perform MaxEntIrl

        auto maxEntIRL = new MaxEntIRL_exact(model, lr, tolerance, weights);
        
        double [] found_weights = maxEntIRL.solve (trajectories, iter % 2 == 0); // alternate solvers


        double err = calcInverseLearningError(model, new LinearReward(features, weights), new LinearReward(features, found_weights), value_error, sizeX * sizeY * 10);

        assert(approxEqual(err, 0, tolerance), "MaxEntIRL found bad solution (err: " ~ to!string(err) ~ ", " ~  to!string(iter) ~ ") : " ~ to!string(found_weights) ~ " correct: " ~ to!string(weights));
    }    

}


