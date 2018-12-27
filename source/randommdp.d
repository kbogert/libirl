module randommdp;

import discretemdp;
import discretefunctions;
import std.conv;
import std.typecons;
import std.random;

class RandomMDPState : discretemdp.State {
     private int id;
     private bool terminal;

     public this(int ID, bool term = false) {
          id = ID;
          terminal = term;
     }

     public int getID() {
          return id;
     }

     public override string toString() {
          return "State: " ~ to!string(id);
     }

     override bool opEquals(Object o) {
          auto rhs = cast(RandomMDPState)o;
          if (!rhs) return false;

          return id == rhs.id;
     }

     override size_t toHash() @trusted nothrow {
          return id;
     }

     override bool isTerminal() {
        return terminal;
    }
}

class RandomMDPAction : discretemdp.Action {

    private int id;
    
    public this (int id) {
        this.id = id;
    }
    
    public int getID() {
        return id;
    }
     
    public override string toString() {
        return "Action: " ~ to!string(id);
    }

    override bool opEquals(Object o) {
        auto rhs = cast(RandomMDPAction)o;
        if (!rhs) return false;

        return id == rhs.id;
    }

    override size_t toHash() @trusted nothrow {
        return id;
    }

    override State getIdealStateFor(State s) {
        return null;
    }
    
}


class RandomMDPStateSpace : discretefunctions.Set!(State) {

    public this(int size) {
        Tuple!(State) [] tempArr;
        for (int i = 0; i < size; i ++) {
            tempArr ~= tuple(cast(State)new RandomMDPState(i));
        }

        super(tempArr);
    }

}


class RandomMDPActionSpace : discretefunctions.Set!(Action) {

    public this(int size) {

        Tuple!(Action) [] tempArr;

        for (int i = 0; i < size; i ++) {
            tempArr ~= tuple(cast(Action)new RandomMDPAction(i));
        }
        
        super(tempArr);
    }


}

public BasicModel generateRandomMDP (int state_count, int action_count, double transition_exponent,
                                     double weight_scale, double gamma, out UniqueFeaturesPerStateActionReward reward_obj) {

    RandomMDPStateSpace states = new RandomMDPStateSpace(state_count);
    RandomMDPActionSpace actions = new RandomMDPActionSpace(action_count);


    auto transitions = new ConditionalDistribution!(State, State, Action)(states, states.cartesian_product(actions));
    foreach (s ; states) {
        foreach (a ; actions) {
            transitions[s[0], a[0]] = new Distribution!(State)(states, DistInitType.Exponential, transition_exponent);
        }
    }
    
    double [] true_weights = new double[state_count * action_count];
    foreach (ref w ; true_weights) {
        w = uniform(-1.0 * weight_scale, 1.0 * weight_scale);
    }
    reward_obj = new UniqueFeaturesPerStateActionReward(states, actions, true_weights);

    return new BasicModel(states, actions, transitions, reward_obj.toFunction(), gamma, new Distribution!(State)(states, DistInitType.Uniform), 0.0001 * weight_scale);

}

