module gridworldmdptest;

import gridworld;
import discretemdp;
import discretefunctions;
import tested;
import std.stdio;
import std.conv;
import std.typecons;
import std.math;

@name("Deterministic Gridworld building")
unittest {

    int sizeX = 10;
    int sizeY = 10;
    double gamma = 0.95;
    double value_error = 0.001;

    auto optimal_state = new GridWorldState(sizeX - 1, sizeY - 1);
    auto optimal_action = new GridWorldAction(1, 0) ;    

    GridWorldStateSpace states = new GridWorldStateSpace(sizeX, sizeY);
    GridWorldActionSpace actions = new GridWorldActionSpace();

    Function!(double [], State, Action) features = new Function!(double [], State, Action)(states.cartesian_product(actions), [0]);

    foreach (a ; actions) {
        features[ optimal_state , a[0] ] = [1.0];
    }

    auto lr = new LinearReward(features, [1.0]);

    auto transitions = new ConditionalDistribution!(State, State, Action)(states, states.cartesian_product(actions));

    foreach (s ; states) {
        foreach (a ; actions) {

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

    auto model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform), value_error * max ( max( lr.toFunction())) );

    auto V = value_iteration(model, value_error * max ( max( lr.toFunction())) );

    double optimal_value = 0;


    foreach (i ; 0 .. 10000) {
        optimal_value += pow(gamma, i) * lr[optimal_state, optimal_action];
    }

    assert( abs (V[optimal_state] - optimal_value) <= value_error, "Incorrect optimal value, " ~ to!string(V[optimal_state]) ~ " correct: " ~ to!string(optimal_value) );

    version(fullunittest) {
    
        // increase accuracy (decrease error)

        gamma = 0.99;
        value_error = 0.0001;

        model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform),  value_error * max ( max( lr.toFunction())));
        V = value_iteration(model, value_error * max ( max( lr.toFunction())) );

        optimal_value = 0;
        foreach (i ; 0 .. 10000) {
            optimal_value += pow(gamma, i) * lr[optimal_state, optimal_action];
        }

        assert( abs (V[optimal_state] - optimal_value) <= value_error, "Incorrect optimal value 2, " ~ to!string(V[optimal_state]) ~ " correct: " ~ to!string(optimal_value) );
    
    }
        
}


@name("Gridworld simulation with non-terminal state")
unittest {

    int sizeX = 10;
    int sizeY = 10;
    double gamma = 0.95;
    double value_error = 0.001;

    auto optimal_state = new GridWorldState(sizeX - 1, sizeY - 1);
    auto optimal_action = new GridWorldAction(1, 0) ;    

    GridWorldStateSpace states = new GridWorldStateSpace(sizeX, sizeY);
    GridWorldActionSpace actions = new GridWorldActionSpace();

    Function!(double [], State, Action) features = new Function!(double [], State, Action)(states.cartesian_product(actions), [0]);

    foreach (a ; actions) {
        features[ optimal_state , a[0] ] = [1.0];
    }

    auto lr = new LinearReward(features, [1.0]);

    auto transitions = new ConditionalDistribution!(State, State, Action)(states, states.cartesian_product(actions));

    foreach (s ; states) {
        foreach (a ; actions) {

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

    auto model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform), value_error * max ( max( lr.toFunction())));

    auto V = value_iteration(model, value_error * max ( max( lr.toFunction())) );

    foreach (i ; 0 .. (sizeX * sizeY) ) {
        auto trajectory = simulate(model, to_stochastic_policy(optimum_policy(V, model), actions), sizeX + sizeY, model.initialStateDistribution()  );


        assert(trajectory[$][0] == optimal_state, "Agent did not reach the optimal state");
    }
    
}


@name("Gridworld simulation with terminal state")
unittest {

    int sizeX = 10;
    int sizeY = 10;
    double gamma = 0.95;
    double value_error = 0.001;

    auto optimal_state = new GridWorldState(sizeX - 1, sizeY - 1);
    auto optimal_action = new GridWorldAction(1, 0) ;    

    GridWorldStateSpaceWithTerminal states = new GridWorldStateSpaceWithTerminal(sizeX, sizeY);
    GridWorldActionSpace actions = new GridWorldActionSpace();

    Function!(double [], State, Action) features = new Function!(double [], State, Action)(states.cartesian_product(actions), [0]);

    foreach (a ; actions) {
        features[ optimal_state , a[0] ] = [1.0];
    }

    auto lr = new LinearReward(features, [1.0]);

    auto transitions = new ConditionalDistribution!(State, State, Action)(states, states.cartesian_product(actions));

    foreach (s ; states) {

        foreach (a ; actions) {

            auto newState = (cast(GridWorldAction)a[0]).getIdealStateFor(cast(GridWorldState)s[0]);

            Distribution!State ds = new Distribution!(State)(states, 0.0);

            if (s[0] != optimal_state) {

                if (states.contains(cast(State)newState)) {
                    ds[newState] = 1.0;
                } else {
                    ds[s[0]] = 1.0;
                }

                ds.normalize();

            }    
            
            transitions[s[0], a[0]] = ds;
        }
    }

    auto model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform), value_error * max ( max( lr.toFunction())));

    auto V = value_iteration(model, value_error * max ( max( lr.toFunction())) );

    assert( V[optimal_state] == lr[optimal_state, optimal_action], "Incorrect optimal value with terminal state, " ~ to!string(V[optimal_state]) ~ " correct: " ~ to!string(lr[optimal_state, optimal_action]) );


    foreach (i ; 0 .. (sizeX * sizeY) ) {
        auto trajectory = simulate(model, to_stochastic_policy(optimum_policy(V, model), actions), sizeX + sizeY, model.initialStateDistribution()  );
        
        assert(trajectory[$][0] == optimal_state, "Agent did not reach the optimal state");

        if (trajectory.length() > 1) {

            assert(trajectory[$ - 1][0] != optimal_state, "Terminal state is not functioning correctly with simulate");

        }
        
    }
        
}



@name("Deterministic Transition function test")
unittest {

    int sizeX = 10;
    int sizeY = 10;
    double gamma = 0.95;
    double value_error = 0.001;
    double idealStateTransitionProb = 1.0;

    auto optimal_state = new GridWorldState(sizeX - 1, sizeY - 1);
    auto optimal_action = new GridWorldAction(1, 0) ;    

    GridWorldStateSpace states = new GridWorldStateSpace(sizeX, sizeY);
    GridWorldActionSpace actions = new GridWorldActionSpace();

    Function!(double [], State, Action) features = new Function!(double [], State, Action)(states.cartesian_product(actions), [0]);

    foreach (a ; actions) {
        features[ optimal_state , a[0] ] = [1.0];
    }

    auto lr = new LinearReward(features, [1.0]);

    auto transitions = build_simple_transition_function(states, actions, idealStateTransitionProb, & otherActionsErrorFunction);

    auto model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform), value_error * max ( max( lr.toFunction())));

    auto V = value_iteration(model, value_error * max ( max( lr.toFunction())) );

    double optimal_value = 0;


    foreach (i ; 0 .. 10000) {
        optimal_value += pow(gamma, i) * lr[optimal_state, optimal_action];
    }

    assert( abs (V[optimal_state] - optimal_value) <= value_error, "Incorrect optimal value, " ~ to!string(V[optimal_state]) ~ " correct: " ~ to!string(optimal_value) );

    
    foreach (i ; 0 .. (sizeX * sizeY) ) {
        auto trajectory = simulate(model, to_stochastic_policy(optimum_policy(V, model), actions), sizeX + sizeY, model.initialStateDistribution()  );


        assert(trajectory[$][0] == optimal_state, "Agent did not reach the optimal state");
    }
    
    version(fullunittest) {
    
        // increase accuracy (decrease error)

        gamma = 0.99;
        value_error = 0.0001;

        model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform),  value_error * max ( max( lr.toFunction())));
        V = value_iteration(model, value_error * max ( max( lr.toFunction())) );

        optimal_value = 0;
        foreach (i ; 0 .. 10000) {
            optimal_value += pow(gamma, i) * lr[optimal_state, optimal_action];
        }

        assert( abs (V[optimal_state] - optimal_value) <= value_error, "Incorrect optimal value 2, " ~ to!string(V[optimal_state]) ~ " correct: " ~ to!string(optimal_value) );
    
    }
        
}


@name("Stochastic Gridworld building")
unittest {

    int sizeX = 10;
    int sizeY = 10;
    double gamma = 0.95;
    double value_error = 0.001;
    double idealStateTransitionProb = .90;

    auto optimal_state = new GridWorldState(sizeX - 1, sizeY - 1);
    auto optimal_action = new GridWorldAction(1, 0) ;    

    GridWorldStateSpaceWithTerminal states = new GridWorldStateSpaceWithTerminal(sizeX, sizeY);
    GridWorldActionSpace actions = new GridWorldActionSpace();

    Function!(double [], State, Action) features = new Function!(double [], State, Action)(states.cartesian_product(actions), [0]);

    foreach (a ; actions) {
        features[ optimal_state , a[0] ] = [1.0];
    }

    auto lr = new LinearReward(features, [1.0]);

    auto transitions = build_simple_transition_function(states, actions, idealStateTransitionProb, & otherActionsErrorFunction);

    assert( transitions[ tuple(cast(State)new GridWorldState(3, 3), cast(Action)new GridWorldAction(1, 0)) ][new GridWorldState(4,3)] == idealStateTransitionProb, "Transition function not constructed correctly");

    auto model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform), value_error * max ( max( lr.toFunction())));

    auto V = value_iteration(model, value_error * max ( max( lr.toFunction())) );

    double optimal_value = 0;


    foreach (i ; 0 .. 10000) {
        optimal_value += pow(gamma, i) * lr[optimal_state, optimal_action];
    }

    assert( abs (V[optimal_state] - optimal_value) <= value_error, "Incorrect optimal value, " ~ to!string(V[optimal_state]) ~ " correct: " ~ to!string(optimal_value) );

    
    foreach (i ; 0 .. (sizeX * sizeY) ) {
        auto trajectory = simulate(model, to_stochastic_policy(optimum_policy(V, model), actions), sizeX * sizeX, model.initialStateDistribution()  );

        assert(trajectory[$][0] == optimal_state, "Agent did not reach the optimal state");
    }
    
    version(fullunittest) {
    
        // increase accuracy (decrease error)

        gamma = 0.99;
        value_error = 0.0001;

        model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform),  value_error * max ( max( lr.toFunction())));
        V = value_iteration(model, value_error * max ( max( lr.toFunction())) );

        optimal_value = 0;
        foreach (i ; 0 .. 10000) {
            optimal_value += pow(gamma, i) * lr[optimal_state, optimal_action];
        }

        assert( abs (V[optimal_state] - optimal_value) <= value_error, "Incorrect optimal value 2, " ~ to!string(V[optimal_state]) ~ " correct: " ~ to!string(optimal_value) );
    
    }
        
}

@name("State Visitation Frequency test")
unittest {

    int sizeX = 10;
    int sizeY = 10;
    double gamma = 0.95;
    double value_error = 0.001;
    double idealStateTransitionProb = .90;
    size_t numTrajectories = 200000;
    int trajLength = cast(int)ceil(log(value_error / 2) / log(gamma));
    
    auto optimal_state = new GridWorldState(sizeX - 1, sizeY - 1);
    auto optimal_action = new GridWorldAction(1, 0) ;    

    GridWorldStateSpaceWithTerminal states = new GridWorldStateSpaceWithTerminal(sizeX, sizeY);
    GridWorldActionSpace actions = new GridWorldActionSpace();

    Function!(double [], State, Action) features = new Function!(double [], State, Action)(states.cartesian_product(actions), [0]);

    foreach (a ; actions) {
        features[ optimal_state , a[0] ] = [1.0];
    }

    auto lr = new LinearReward(features, [1.0]);

    auto transitions = build_simple_transition_function(states, actions, idealStateTransitionProb, & otherActionsErrorFunction);

    assert( transitions[ tuple(cast(State)new GridWorldState(3, 3), cast(Action)new GridWorldAction(1, 0)) ][new GridWorldState(4,3)] == idealStateTransitionProb, "Transition function not constructed correctly");

    auto model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform), value_error);

    auto V = value_iteration(model, value_error * max ( max( lr.toFunction())) );
    auto pi = optimum_policy(V, model);

    // find empirical state visitation frequency

    Function!(double, State) mu2 = new Function!(double, State)(model.S(), 0.0);

    foreach(i; 0 .. numTrajectories) {

        auto traj = simulate(model, to_stochastic_policy(pi, model.A()), trajLength, model.initialStateDistribution(), true );

        auto g = 1.0;
        
        foreach(timestep; traj) {

            mu2[timestep[0]] += g / numTrajectories;
            g *= gamma;
        }
    }
    
    
    auto mu1 = stateVisitationFrequency(model, pi, value_error);

    foreach (s; model.S()) {
        assert (isClose(mu1[s[0]], mu2[s[0]], 0.01, 1e-5), "State Visitation Frequency functions differ: 1: " ~ to!string(mu1[s[0]]) ~ "\n\n 2: " ~ to!string(mu2[s[0]]) ~ " " ~ to!string(s[0]));
    }    
        

    // test sub-rational cases


    auto model2 = new SoftMaxModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform), value_error);

    auto V2 = soft_max_value_iteration(model2, value_error * max ( max( lr.toFunction())) );
    auto pi2 = soft_max_policy(V2, model2);

    // find empirical state visitation frequency

    mu2 = new Function!(double, State)(model2.S(), 0.0);

    foreach(i; 0 .. numTrajectories) {

        auto traj = simulate(model2, pi2, trajLength, model2.initialStateDistribution(), true );

        auto g = 1.0;
        
        foreach(timestep; traj) {

            mu2[timestep[0]] += g / numTrajectories;
            g *= gamma;
        }
    }
    
    
    mu1 = stateVisitationFrequency(model2, pi2, value_error);

    foreach (s; model2.S()) {
        assert (isClose(mu1[s[0]], mu2[s[0]], 0.01, 1e-5), "State Visitation Frequency functions differ: 1: " ~ to!string(mu1[s[0]]) ~ "\n\n 2: " ~ to!string(mu2[s[0]]) ~ " " ~ to!string(s[0]));
    }    
           
}
