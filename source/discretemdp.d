module discretemdp;

import mdp;
import discretefunctions;
import std.math;
import std.typecons;
import std.numeric;
import std.conv;
import std.algorithm.comparison;
import std.random;

class State {


    abstract bool isTerminal();
}

class Action {

    abstract State getIdealStateFor (State s);

}

class Reward {

    abstract double opIndex(State s, Action a );

    abstract Function!(double, State, Action) toFunction();

}


class Model {

    abstract Set!(State) S();

    abstract Set!(Action) A();

    abstract ConditionalDistribution!(State, State, Action) T();

    abstract Function!(double, State, Action) R();

    abstract double gamma();

    abstract Distribution!(State) initialStateDistribution();
}


public Function!(double, State) value_iteration(Model m, double tolerance, int max_iter = int.max) {

    Function!(double, State) v_next;
    Function!(double, State) v_prev = max( m.R() );
    auto T = m.T().flatten();
    
    double diff = max( v_prev );
    int iter = 0;

    while (diff > tolerance*(1 - m.gamma()) / m.gamma() && iter < max_iter) {
        
        v_next = max( m.R() + m.gamma() * sumout!(State)( T * v_prev ) ) ;

        diff = max ( v_next - v_prev ); 
        v_prev = v_next;
    }

    return v_next;
}

public Function!(double, State, Action) q_value_iteration(Model m, double tolerance, int max_iter = int.max) {

    return sumout!(State) (m.T() * value_iteration(m, tolerance, max_iter) );
}

public Function!(Tuple!(Action), State) optimum_policy (Function!(double, State) V, Model m) {

    return argmax( sumout!(State)( m.T() * V ) );

}

public Function!(Tuple!(Action), State) optimum_policy (Function!(double, State, Action) Q) {

    return argmax( Q );

}

public Function!(Tuple!(Action), State) optimum_policy (Model m, double tolerance, int max_iter = int.max) {

    return argmax (q_value_iteration (m, tolerance, max_iter) );
}


public ConditionalDistribution!(Action, State) to_stochastic_policy(Function!(Tuple!(Action), State) policy, Set!Action action_set) {

        auto returnval = new ConditionalDistribution!(Action, State)(action_set, policy.param_set());

        foreach( s ; policy.param_set()) {

            auto d = new Distribution!(Action)(action_set);

            d [ policy [s[0]] ] = 1.0;

            returnval[ s[0] ] = d;
        }

        return returnval;    
}

public Sequence!(State, Action) simulate(Model m, ConditionalDistribution!(Action, State) stochastic_policy, int timesteps, Distribution!(State) initialStates) {

    auto s = initialStates.sample();

    auto returnval = new Sequence!(State, Action)(timesteps);

    size_t cur_timestep = 0;

    while (cur_timestep < timesteps) {

        if (s[0].isTerminal()) {

            returnval[cur_timestep] = tuple!(State, Action)(s[0], null);
            returnval.setLength(cur_timestep + 1);
            break;
            
        }
        
        auto a = stochastic_policy[s].sample();

        auto sa = tuple(s[0], a[0]);

        returnval[cur_timestep] = sa;

        s = m.T()[ sa ].sample();
        
        cur_timestep ++;
    }
    
    
    return returnval;

}

public ConditionalDistribution!(State, State, Action) build_simple_transition_function(Set!(State) states, Set!(Action) actions, double ideal_state_prob, Function!(double, State) function (Set!(State) all_states, Set!(Action) all_actions, State state, Action action, State ideal_state, double remaining_prob_mass ) error_function, bool zero_out_terminal_transitions = false ) {

    ideal_state_prob = min(1.0, max(0.0, ideal_state_prob)) ;


    auto returnval = new ConditionalDistribution!(State, State, Action)(states, states.cartesian_product(actions));

    foreach ( s ; states) {

        // handle reaching a terminal state 
        if (s[0].isTerminal()) {

            foreach ( a ; actions ) {
                // terminals really actually terminate execution, no transitions from them
                if (zero_out_terminal_transitions) {

                    returnval[tuple( s[0], a[0])] = new Distribution!(State)(states, 0.0);
                
                } else {
                    // terminals are black holes, once in them the agent cannot escape

                    auto temp = new Distribution!(State)(states, 0.0);

                    temp[tuple(s[0])] = 1.0;
                    
                    returnval[tuple( s[0], a[0])] = temp;
                
                }

            }

            continue;
        }



        foreach ( a ; actions ) {

            auto total_transitions = new Function!(double, State)(states, 0.0);


            // ideal state
            
            auto ideal_state = a[0].getIdealStateFor( s[0] );

            if ( ! states.contains(ideal_state) ) {
                ideal_state = s[0];
            }

            total_transitions[tuple(ideal_state)] = ideal_state_prob;

            // distribute leftover prob mass according to error function

            total_transitions = total_transitions + error_function(states, actions, s[0], a[0], ideal_state, 1.0 - ideal_state_prob);

            auto totalMass = sumout( total_transitions );


            
            if (totalMass < 1.0) {
                // we've still got some prob mass left, distribute to all states

                foreach (s_prime ; states) {

                    total_transitions[tuple(s_prime[0])] += (1.0 - totalMass) / states.size();
                }
                
            
            }


            returnval[tuple(s[0], a[0])] = new Distribution!(State)(total_transitions);
            
        }

    }

    return returnval;
}

// distribute the remaining probability mass equally among the ideal states for all other actions
Function!(double, State) otherActionsErrorFunction (Set!(State) allStates, Set!(Action) allActions, State state, Action action, State ideal_state, double remainingProbMass ) {

    auto returnval = new Function!(double, State)(allStates, 0.0); 

    auto amount = remainingProbMass / (allActions.size() - 1);
    
    foreach( a ; allActions ) {

        if (a[0] != action) {

            auto nextState = a[0].getIdealStateFor( state );

            if ( ! allStates.contains(nextState) ) {
                nextState = state;
            }

            returnval[tuple(nextState)] += amount;
        }
                
    }    

    return returnval;
}

// distribute the remaining probability mass equally among all other states than the ideal
Function!(double, State) allOtherStatesErrorFunction (Set!(State) allStates, Set!(Action) allActions, State state, Action action, State ideal_state, double remainingProbMass ) {
    
    auto returnval = new Function!(double, State)(allStates, 0.0); 

    auto amount = remainingProbMass / (allStates.size() - 1);


    foreach (s ; allStates) {

        if (s[0] != ideal_state) {

            returnval[tuple(s[0])] = amount;
        }

    }
    
    return returnval;
}


class BasicModel : Model {

    protected Set!(State) states;
    protected Set!(Action) actions;
    protected ConditionalDistribution!(State, State, Action) transitions;
    protected Function!(double, State, Action) rewards;
    protected double gam;
    protected Distribution!(State) isd;
    
    public this(Set!(State) states, Set!(Action) actions, ConditionalDistribution!(State, State, Action) transitions, Function!(double, State, Action) rewards, double gamma, Distribution!(State) initialStateDistribution) {
        this.states = states;
        this.actions = actions;
        this.transitions = transitions;
        this.rewards = rewards;
        this.gam = gamma;
        this.isd = initialStateDistribution;
    }
    
    public override Set!(State) S() {
        return states;
    }

    public override Set!(Action) A() {
        return actions;
    }

    public override ConditionalDistribution!(State, State, Action) T() {
        return transitions;
    }

    public override Function!(double, State, Action) R() {
        return rewards;
    }

    public override double gamma() {
        return gam;
    }

    public override Distribution!(State) initialStateDistribution() {
        return isd;
    }
}


class LinearReward : Reward {

    protected double [] weights;
    protected Function!(double [], State, Action) features;
    protected size_t size;

    public this(Function!(double [], State, Action) f) {
        features = f;
        foreach(key ; f.param_set()) {
            size = f[key].length;
            break;
        }
    }

    public this(Function!(double [], State, Action) f, double [] weights) {
        this(f);
        setWeights(weights);
    }

    public size_t getSize() {
        return size;
    }
        
    public double [] getWeights() {
        return weights;
    }

    public void setWeights(double [] w) {
        if (w.length != size)
            throw new Exception("Incorrect weight size, the feature function is of size: " ~ to!string(size));
            
        weights = w;
    }

    public double [] getFeatures(State s, Action a) {
        return features[tuple(s, a)];
    }

    public override double opIndex(State s, Action a) {
        return dotProduct(weights, getFeatures(s, a));
    }
        
    public double opIndex(Tuple!(State, Action) t) {
        return dotProduct(weights, features[t]);
    }

    public override Function!(double, State, Action) toFunction() {

        auto returnval = new Function!(double, State, Action)(features.param_set(), 0.0);

        foreach (key; features.param_set()) {
            returnval[key] = opIndex(key);
        }

        return returnval;
    }
}

class RandomStateReward : Reward {

    protected double[State] rewards;
    protected Set!(State) stateSet;
    protected Set!(Action) actionSet;
    
    public this(Set!(State) states, Set!(Action) actions, double scale = 1.0) {
        foreach(s ; states) {
            rewards[s[0]] = uniform( -1.0 * scale, 1.0 * scale);
        }
        
        stateSet = states;
        actionSet = actions;
    }

    public override double opIndex(State s, Action a) {
        return rewards[s];
    }
        
    public double opIndex(Tuple!(State, Action) t) {
        return rewards[t[0]];
    }

    public override Function!(double, State, Action) toFunction() {

        auto returnval = new Function!(double, State, Action)(stateSet.cartesian_product(actionSet), 0.0);

        foreach (key; returnval.param_set()) {
            returnval[key] = opIndex(key);
        }

        return returnval;
    }
}

/*

    Need softmax versions of q-value iteration and value iteration

*/


public Function!(double, State) soft_max_value_iteration(Model m, double tolerance, int max_iter = int.max) {

    Function!(double, State) v_next;
    Function!(double, State) v_prev = softmax( m.R() );
    auto T = m.T().flatten();
    
    double diff = max( v_prev );
    int iter = 0;

    while (diff > tolerance*(1 - m.gamma()) / m.gamma() && iter < max_iter) {
        
        v_next = softmax( m.R() + m.gamma() * sumout!(State)( T * v_prev ) ) ;

        diff = max ( v_next - v_prev ); 
        v_prev = v_next;
    }

    return v_next;
}



public Function!(double, State, Action) soft_max_q_value_iteration(Model m, double tolerance, int max_iter = int.max) {

    return sumout!(State) (m.T() * soft_max_value_iteration(m, tolerance, max_iter) );
}

public ConditionalDistribution!(Action, State) soft_max_policy(Function!(double, State) V, Model m) {

    auto Q = sumout!(State)( m.T() * V);

    return soft_max_policy( Q , m );

}

public ConditionalDistribution!(Action, State) soft_max_policy(Function!(double, State, Action) Q, Model m) {

    auto qmax = max(Q);
    
    auto returnval = new ConditionalDistribution!(Action, State)(m.A(), m.S());

    foreach( s ; m.S() ) {

        auto d = new Distribution!(Action)(m.A());

        auto smax = qmax[s];
        double q_total = 0;
        
        foreach (a ; m.A()) {
            q_total += exp( Q[tuple(s[0],a[0])] - smax );
        }
        if (q_total == 0) {
            q_total = double.min_normal;
        }
        q_total = smax + log( q_total );
        
        foreach (a ; m.A()) {
            d [a] = exp( Q[tuple(s[0],a[0])] ) / q_total;
        }
        
        returnval[ s ] = d;
    }

    return returnval;
    
}

