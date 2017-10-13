import discretemdp;



class GridWorldStateSpace : DiscreteStateSpace {

     private State [] states;


     public State [] getStates() {
          return states;
     }

}


class GridWorldActionSpace : DiscreteActionSpace {

}


class GridWorldModel : DiscreteModel {


     public this(int xSpace, int ySpace, double transitionProb) {


          // how to connect the transitions with the state and action objects?
          // easiest thing to do is just have states store their location, and actions store their effect
          // need the ability to print out states and actions
     }

}

class GridWorldReward : LinearReward {


}


