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

     abstract public size_t size();
     abstract public bool contains(T i);
     abstract int opApply(int delegate(ref T) dg);
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

     public this(Space!T s, mdp.DistInitType init = mdp.DistInitType.None) {
          mySpace = s;
          normalized = false;
          final switch(init) {
              case mdp.DistInitType.Uniform:
                  foreach(T key ; mySpace) {
                       myDistribution[key] = 1.0;
                  }
                  myDistribution.rehash();
                  break;
              case mdp.DistInitType.RandomFromUniform:
                  import std.random;
                  foreach(T key ; mySpace) {
                       myDistribution[key] = uniform01();
                  }
                  myDistribution.rehash();
                  break;
              case mdp.DistInitType.RandomFromGaussian:
                  import std.random;
                  foreach(T key ; mySpace) {
                       double total = 0;
                       for (int i = 0; i < 12; i ++)  // irwin-hall approximation of the normal distribution 
                            total += uniform(-1.0, 1.0);
                       myDistribution[key] = total;
                  }
                  myDistribution.rehash();
                  break;
              case mdp.DistInitType.None:
                  break;

          }
     }


     public void normalize() {
          if (normalized) return;

          auto tot = 0.0;
          foreach(val ; myDistribution.values) {
               tot += val;
          }

          if (tot == 0.0) {
               throw new Exception("Empty distribution or all zero probabilities, cannot normalize");
          }

          foreach( key ; myDistribution.keys) {
               myDistribution[key] /= tot;
          }

          normalized = true;
     }

     public bool isNormalized() {
          return normalized;
     }

     // will always return a sample
     public T sample() {
          normalize();

          import std.random;

          auto rand = uniform(0.0, 1.0);

          auto keys = myDistribution.keys;
          randomShuffle(keys);

          auto mass = 0.0;
          foreach (T k; keys) {
               mass += myDistribution[k];

               if (mass >= rand)
                    return k;
          }

          debug {
               import std.conv;
               throw new Exception("Didn't find a key to sample, ended at: " ~ to!string(mass) ~ " but wanted " ~ to!string(rand));
          } else {
                return keys[$-1];
          }

     }

     public T argmax() {

          auto max = -double.max;
          T returnval = null;

          foreach(key, val ; myDistribution) {
                if (val > max) {
                     max = val;
                     returnval = key;
                }
          }

          return returnval;

     }

     override public string toString() {
          import std.conv;
          string returnval = "";

          foreach(key, val ; myDistribution) {
                returnval ~= key.toString() ~ " => " ~ to!string(val) ~ "\n";
          }
          return returnval;
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
          if ( mySpace !is null && ! mySpace.contains(i)) {
               throw new Exception("ERROR, key is not in the space this distribution is defined over.");
          }
          myDistribution[i] = value;
          normalized = false;
     }

     size_t opDollar(size_t pos)() {
          return size();
     }

     size_t size() {
          return myDistribution.length;
     }

     void opIndexOpAssign(string op)(double rhs, T key) {
          double* p;
          p = (key in myDistribution);
          if (p is null) {
               if ( mySpace !is null && ! mySpace.contains(key)) {
                    throw new Exception("ERROR, key is not in the space this distribution is defined over.");
               }
               myDistribution[key] = 0;
               p = (key in myDistribution);
          }
          mixin("*p " ~ op ~ " rhs");

          normalized = false;
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

     int opApply(int delegate(ref T, ref double) dg) {
          int result = 0;
          foreach (key, value ; myDistribution) {
               result = dg(key, value);
               if (result) break;

          }
          return result;
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
