module gridworld;


import discretemdp;
import discretefunctions;
import std.conv;
import std.typecons;


class GridWorldState : discretemdp.State {
     private int x;
     private int y;

     public this(int X, int Y) {
          x = X;
          y = Y;
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
        return false;
    }
}

class GridWorldAction : discretemdp.Action {

    int xMod;
    int yMod;

    public this (int x, int y) {
        xMod = x;
        yMod = y;
    }

    public GridWorldState apply(GridWorldState start) {
        return new GridWorldState(start.getX() + xMod, start.getY() + yMod);
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

    GridWorldStateSpace states = new GridWorldStateSpace(sizeX, sizeY);
    GridWorldActionSpace actions = new GridWorldActionSpace();

    Function!(double [], State, Action) features = new Function!(double [], State, Action)(states.cartesian_product(actions), [0]);

    foreach (a ; actions) {
        features[ new GridWorldState(9,9) , a[0] ] = [1.0];
    }

    auto lr = new LinearReward(features, [1.0]);

    auto transitions = new ConditionalDistribution!(State, State, Action)(states, states.cartesian_product(actions));

    foreach (s ; states) {
        foreach (a ; actions) {

            auto newState = (cast(GridWorldAction)a[0]).apply(cast(GridWorldState)s[0]);

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

    auto model = new BasicModel(states, actions, transitions, lr.toFunction(), 0.95, new Distribution!(State)(states, DistInitType.Uniform));

    auto V = value_iteration(model, 0.1);

    writeln(V);
}


@name("Gridworld simulation with terminal state")
unittest {
}

@name("Gridworld simulation with non-terminal state")
unittest {
}
