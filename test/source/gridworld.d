module gridworld;


import discretemdp;
import std.conv;

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

     override int opCmp(Object o) const {
          return 0;
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



class GridWorldStateSpace : discretemdp.StateSpace {

     private GridWorldState [] states;


     public GridWorldState [] getStates() {
          return states;
     }

     override public ulong size() {
          return states.length;
     }
}


class GridWorldActionSpace : discretemdp.ActionSpace {

     override public ulong size() {
          return 0;
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
