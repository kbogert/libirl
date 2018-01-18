module gridworld;


import discretemdp;
import discretefunctions;
import std.conv;
import std.typecons;
import std.math;

class GridWorldState : discretemdp.State {
     private int x;
     private int y;
     private bool terminal;

     public this(int X, int Y, bool term = false) {
          x = X;
          y = Y;
          terminal = term;
     }

     public int getX() {
          return x;
     }

     public int getY() {
          return y;
     }

     public override string toString() {
          return "State: " ~ to!string(x) ~ " x " ~ to!string(y);
     }

     override bool opEquals(Object o) {
          auto rhs = cast(GridWorldState)o;
          if (!rhs) return false;

          return x == rhs.x && y == rhs.y;
     }

     override size_t toHash() @trusted nothrow {
          return x * y;
     }

     override bool isTerminal() {
        return terminal;
    }
}

class GridWorldAction : discretemdp.Action {

    int xMod;
    int yMod;

    public this (int x, int y) {
        xMod = x;
        yMod = y;
    }
    
    public override string toString() {
        return "Action: " ~ to!string(xMod) ~ " x " ~ to!string(yMod);
    }

    override bool opEquals(Object o) {
        auto rhs = cast(GridWorldAction)o;
        if (!rhs) return false;

        return xMod == rhs.xMod && yMod == rhs.yMod;
    }

    override size_t toHash() @trusted nothrow {
        return xMod * yMod;
    }

    override GridWorldState getIdealStateFor(State s) {
        auto state = cast(GridWorldState)s;

        if (!state)
            return null;

        return new GridWorldState(state.getX() + xMod, state.getY() + yMod);
    }
    
}


class GridWorldStateSpace : discretefunctions.Set!(State) {


    public this(int sizeX, int sizeY) {
        Tuple!(State) [] tempArr;
        for (int i = 0; i < sizeX; i ++) {
            for (int j = 0; j < sizeY; j ++) {
                tempArr ~= tuple(cast(State)new GridWorldState(i, j));
            }
        } 

        super(tempArr);
    }

}


class GridWorldActionSpace : discretefunctions.Set!(Action) {

    public this() {

        Tuple!(Action) [] tempArr;

        tempArr ~= tuple(cast(Action)new GridWorldAction(1, 0));
        tempArr ~= tuple(cast(Action)new GridWorldAction(-1, 0));
        tempArr ~= tuple(cast(Action)new GridWorldAction(0, -1));
        tempArr ~= tuple(cast(Action)new GridWorldAction(0, 1));
        
        
        super(tempArr);
    }


}


import tested;
import std.stdio;

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

    auto model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform));

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

        model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform));
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

    auto model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform));

    auto V = value_iteration(model, value_error * max ( max( lr.toFunction())) );

    foreach (i ; 0 .. (sizeX * sizeY) ) {
        auto trajectory = simulate(model, to_stochastic_policy(optimum_policy(V, model), actions), sizeX + sizeY, model.initialStateDistribution()  );


        assert(trajectory[$][0] == optimal_state, "Agent did not reach the optimal state");
    }
    
}

class GridWorldStateSpaceWithTerminal : discretefunctions.Set!(State) {


    public this(int sizeX, int sizeY) {
        Tuple!(State) [] tempArr;
        for (int i = 0; i < sizeX; i ++) {
            for (int j = 0; j < sizeY; j ++) {
                tempArr ~= tuple(cast(State)new GridWorldState(i, j, ( i == sizeX - 1 && j == sizeY - 1)));
            }
        } 

        super(tempArr);
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

    auto model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform));

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

    auto model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform));

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

        model = new BasicModel(states, actions, transitions, lr.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform));
        V = value_iteration(model, value_error * max ( max( lr.toFunction())) );

        optimal_value = 0;
        foreach (i ; 0 .. 10000) {
            optimal_value += pow(gamma, i) * lr[optimal_state, optimal_action];
        }

        assert( abs (V[optimal_state] - optimal_value) <= value_error, "Incorrect optimal value 2, " ~ to!string(V[optimal_state]) ~ " correct: " ~ to!string(optimal_value) );
    
    }
        
}

