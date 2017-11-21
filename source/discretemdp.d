module discretemdp;

import mdp;
import std.math;

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
class Distribution(T) {

     double [T] myDistribution;
     bool normalized;
     Space!T mySpace;

     public this() {
        normalized = false;
     }

     public this(double [T] distribution) {
          myDistribution = distribution;
          normalize();

     }

     public this(Space!T s, mdp.DistInitType init = mdp.DistInitType.None, double skewness = 10) {
          mySpace = s;
          normalized = false;

          if (init != mdp.DistInitType.None) {

              foreach(T key ; mySpace) {
                 final switch(init) {
                  case mdp.DistInitType.Uniform:
                      myDistribution[key] = 1.0;
                      break;
                  case mdp.DistInitType.Exponential:
                      import std.random;
                      myDistribution[key] = exp(uniform01() * skewness);
                      break;
                  case mdp.DistInitType.RandomFromGaussian:
                      import std.random;
                      double total = 0;
                      for (int i = 0; i < 12; i ++)  // irwin-hall approximation of the normal distribution 
                           total += uniform01();
                      myDistribution[key] = total;
                      break;
                  case mdp.DistInitType.None: // should never be here
                      break;

                 }

              }

              myDistribution.rehash();
              normalize();

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

          foreach(key, ref val ; myDistribution) {
               val /= tot;
          }

          normalized = true;
     }

     public bool isNormalized() {
          return normalized;
     }

     // will always return a sample
     public T sample() {

          if (size() == 0) {
             return null;
          }
          
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
          mixin("*p " ~ op ~ "= rhs;");

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

     
    double KLD(Distribution!T other_dist) {
	
    	double returnval = 0;
    	foreach (i, pr; myDistribution) {
    		returnval += pr * log ( pr / other_dist[i]);
    	}
    	return returnval;
	
    }  

    double entropy() {
        double returnval = 0;

        foreach (pr ; myDistribution.values) {
            returnval += pr * log (pr);
        }
        return -returnval;

    }

    double crossEntropy(Distribution!T other_dist) {
        return entropy() + KLD(other_dist);
    }

    void optimize() {
        foreach(key, val ; myDistribution) {
            if (val == 0) {
                myDistribution.remove(key);
            }
        }

        myDistribution.rehash();
    }
}

class Model : mdp.Model {

}


class Reward {


}


class LinearReward : Reward {

}
