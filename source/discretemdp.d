module discretemdp;

import mdp;


class State {

}

class Action {


}

class StateAction {

}

class StateActionState {

}



class Space : mdp.Space {

     abstract public ulong size();
}


class StateSpace : Space {

}


class ActionSpace : Space {


}

class StateActionSpace : Space {

}

class StateActionStateSpace : Space {

}

// Holds a discrete distribution, mapping a space of individual objects to normalized probabilities
class Distribution(T) : mdp.Distribution {

     double [T] myDistribution;

     public this(double [T] distribution) {
          myDistribution = distribution;
          normalize();

     }

     public this(Space s) {

     }


     private void normalize() {

     }


     override public string toString() {


     }

     override double opIndex(T ... i) {

     }

     override void opIndexAssign(double, T ... i) {

     }
     
}

// Holds a discrete mapping from one object type to a distribution over
class Mapping : mdp.Mapping {


}

class Model : mdp.Model {

}


class Reward {


}


class LinearReward : Reward {

}
