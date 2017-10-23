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
     bool normalized;
     Space mySpace;


     public this(double [T] distribution) {
          myDistribution = distribution;
          normalize();

     }

     public this(Space s) {
          mySpace = s;
     }


     public void normalize() {


          normalized = true;
     }

     public bool isNormalized() {
          return normalized;
     }

     public T sample() {

     }

     public T argmax() {

     }

     public T argmin() {

     } 

     override public string toString() {


     }

     double opIndex(T i) {
          return myDistribution[i];
     }

     void opIndexAssign(double value, T i) {
          myDistribution[i] = value;
          normalized = false;
     }

     override int opDollar(size_t pos)() {
          return myDistribution.length;
     }

     
     // need the byKey(), byValue(), and byKeyValue() methods     

     auto byKey() {
          return myDistribution.byKey();
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
