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

class GridWorldActionUp : discretemdp.Action {

     public GridWorldState intendedNextState(GridWorldState s) {
           return new GridWorldState(s.getX(), s.getY() - 1);

     }
}

class GridWorldActionDown : discretemdp.Action {

     public GridWorldState intendedNextState(GridWorldState s) {
           return new GridWorldState(s.getX(), s.getY() + 1);

     }

}

class GridWorldActionLeft : discretemdp.Action {

     public GridWorldState intendedNextState(GridWorldState s) {
           return new GridWorldState(s.getX() - 1, s.getY());

     }

}

class GridWorldActionRight : discretemdp.Action {

     public GridWorldState intendedNextState(GridWorldState s) {
           return new GridWorldState(s.getX() + 1, s.getY());

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


class GridWorldActionSpace : discretefunctions.Set!(Action) {

    public this() {

        Tuple!(Action) [] tempArr;

        tempArr ~= tuple(cast(Action)new GridWorldActionUp());
        tempArr ~= tuple(cast(Action)new GridWorldActionDown());
        tempArr ~= tuple(cast(Action)new GridWorldActionLeft());
        tempArr ~= tuple(cast(Action)new GridWorldActionRight());
        
        
        super(tempArr);
    }


}


class GridWorldModel : discretemdp.Model {


     public this(int xSpace, int ySpace, double transitionProb) {


          // how to connect the transitions with the state and action objects?
          // easiest thing to do is just have states store their location, and actions store their effect
          // need the ability to print out states and actions
     }

}

class GridWorldReward : discretemdp.LinearReward {

        Distribution!State uniform;

}


import tested;

@name("arithmetic")
unittest {
        int i = 3;
        assert(i == 3);
        i *= 2;
        assert(i == 6);
        i += 5;
        assert(i == 11);
}
