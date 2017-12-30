module discretemdp;

import mdp;
import discretefunctions;
import std.math;

class State {


    abstract bool isTerminal();
}

class Action {


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


class LinearReward : Reward {

}

public Function!(double, State) value_iteration(Model m, double tolerance, int max_iter = int.max) {

    Function!(double, State) v_next;
    Function!(double, State) v_prev = max( m.R() );

    double diff = max( v_prev );
    int iter = 0;

    while (diff > tolerance && iter < max_iter) {
        v_next = max( m.R() + m.gamma() * sumout!(State)( m.T() * v_prev ) ) ;

        diff = max ( v_next - v_prev ); 
    }

    return v_next;
}
