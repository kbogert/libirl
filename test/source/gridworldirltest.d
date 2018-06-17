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

import maxentIRL;

@name("MaxEntIRL exact random reward function recovery test")
unittest {

    
    int sizeX = 10;
    int sizeY = 10;
    double gamma = 0.95;
    double value_error = 0.001;
    int samples = 10 * sizeX * sizeY;
    double tolerance = 0.01;

    version(fullunittest) {
        samples = samples * 100;
        tolerance = 0.001;
    }
    
    GridWorldStateSpaceWithTerminal states = new GridWorldStateSpaceWithTerminal(sizeX, sizeY);
    GridWorldActionSpace actions = new GridWorldActionSpace();

    double[] tmpArray;
    tmpArray.length = states.size();
    tmpArray[] = 0.0;

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
        iterations = 1000;
    }

    auto transitions = new ConditionalDistribution!(State, State, Action)(states, states.cartesian_product(actions));

    foreach (s ; states) {
        foreach (a ; actions) {

            if (s[0].isTerminal()) {
                transitions[s[0], a[0]] = new Distribution!(State)(states, 0.0);
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
        weights.length = states.size();
        foreach (ref w ; weights) {
            w = uniform(0.0, 1.0);
        }
        
        auto lr = new LinearReward(features, weights);
        auto model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform));

        
        auto V = soft_max_value_iteration(model, value_error * max ( max( lr.toFunction())) , sizeX * sizeY * 10);
        auto policy = soft_max_policy(V, model);
        
        Sequence!(State, Action) [] trajectories;
        foreach (j ; 0 .. samples ) {
            trajectories ~= simulate(model, policy, sizeX + sizeY, model.initialStateDistribution() );
        }

        // Perform MaxEntIrl

        auto maxEntIRL = new MaxEntIRL_exact(model, lr, tolerance / 2);
        
        double [] found_weights = maxEntIRL.solve (trajectories);

        foreach (j ; 0 .. found_weights.length) {
            assert(approxEqual(weights[j], found_weights[j], tolerance), "MaxEntIRL found bad solution: " ~ to!string(found_weights) ~ " correct: " ~ to!string(weights));
        }
    }    

}


