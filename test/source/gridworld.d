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


class GridWorldStateSpace : discretefunctions.Set!(GridWorldState) {


    public this(int sizeX, int sizeY) {
        Tuple!(GridWorldState) [] tempArr;
        for (int i = 0; i < sizeX; i ++) {
            for (int j = 0; j < sizeY; j ++) {
                tempArr ~= tuple(new GridWorldState(i, j));
            }
        } 

        super(tempArr);
    }

    public this(Set!(GridWorldState) toCopy) {
        super(toCopy.storage.dup);
    }
}


class GridWorldActionSpace : discretefunctions.Set!(GridWorldAction) {

    public this() {

        Tuple!(GridWorldAction) [] tempArr;

        tempArr ~= tuple(new GridWorldAction(1, 0));
        tempArr ~= tuple(new GridWorldAction(-1, 0));
        tempArr ~= tuple(new GridWorldAction(0, -1));
        tempArr ~= tuple(new GridWorldAction(0, 1));
        
        
        super(tempArr);
    }


}


import tested;
import std.stdio;

@name("Gridworld building")
unittest {

    int sizeX = 10;
    int sizeY = 10;

    GridWorldStateSpace states = new GridWorldStateSpace(sizeX, sizeY);
    GridWorldActionSpace actions = new GridWorldActionSpace();

    Function!(double [], GridWorldState, GridWorldAction) features = new Function!(double [], GridWorldState, GridWorldAction)(states.cartesian_product(actions), [0]);

    foreach (a ; actions) {
        features[ new GridWorldState(9,9) , a[0] ] = [1.0];
    }

    auto lr = new LinearReward!(GridWorldState, GridWorldAction)(features, [1.0]);
    

}
