module gridworld;


import discretemdp;
import discretefunctions;
import std.conv;
import std.typecons;


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


