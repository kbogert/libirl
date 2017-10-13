import discretemdp;


class GridWorldState : discretemdp.DiscreteState {
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
     }

     override hash_t toHash() const {
     }

     override bool opEquals(Object o) {
     }

     override int opCmp(Object o) const {
     }

}

class GridWorldActionUp : discretemdp.DiscreteAction {

     public GridWorldState intendedNextState(GridWorldState s) {
           return new GridWorldState(s.getX(), s.getY() - 1);

     }
}

class GridWorldActionDown : discretemdp.DiscreteAction {

     public GridWorldState intendedNextState(GridWorldState s) {
           return new GridWorldState(s.getX(), s.getY() + 1);

     }

}

class GridWorldActionLeft : discretemdp.DiscreteAction {

     public GridWorldState intendedNextState(GridWorldState s) {
           return new GridWorldState(s.getX() - 1, s.getY());

     }

}

class GridWorldActionRight : discretemdp.DiscreteAction {

     public GridWorldState intendedNextState(GridWorldState s) {
           return new GridWorldState(s.getX() + 1, s.getY());

     }

}



class GridWorldStateSpace : discretemdp.DiscreteStateSpace {

     private GridWorldState [] states;


     public State [] getStates() {
          return states;
     }

     public int size() {
          return states.length;
     }
}


class GridWorldActionSpace : discretemdp.DiscreteActionSpace {

}


class GridWorldModel : discretemdp.DiscreteModel {


     public this(int xSpace, int ySpace, double transitionProb) {


          // how to connect the transitions with the state and action objects?
          // easiest thing to do is just have states store their location, and actions store their effect
          // need the ability to print out states and actions
     }

}

class GridWorldReward : discretemdp.LinearReward {


}


