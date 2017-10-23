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



class Space(T) : mdp.Space {

     abstract public ulong size();
     abstract public bool contains(T i);

}


class StateSpace : Space!State {

}


class ActionSpace : Space!Action {


}

class StateActionSpace : Space!StateAction {

}

class StateActionStateSpace : Space!StateActionState {

}

// Holds a discrete distribution, mapping a space of individual objects to normalized probabilities
class Distribution(T) : mdp.Distribution {

     double [T] myDistribution;
     bool normalized;
     Space!T mySpace;


     public this(double [T] distribution) {
          myDistribution = distribution;
          normalize();

     }

     public this(Space!T s) {
          mySpace = s;
          normalized = false;
     }


     public void normalize() {


          normalized = true;
     }

     public bool isNormalized() {
          return normalized;
     }

     public T sample() {
          normalize();

     }

     public T argmax() {

     }

     public T argmin() {

     } 

     override public string toString() {


     }

     double opIndex(T i) {
          double* p;
          p = (i in myDistribution);
          if (p !is null) {
               return *p;
          }
          if ( mySpace !is null && ! mySpace.contains(i)) {
               throw new Exception("ERROR, key is not in the space this distribution is defined over.");
          }
          return 0;
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

     auto byValue() {
          return myDistribution.byValue();
     }

     auto byKeyValue() {
          return myDistribution.byKeyValue();
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
