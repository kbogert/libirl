module randommdptest;

import discretemdp;
import discretefunctions;
import tested;
import std.random;
import randommdp;
import std.typecons;
import std.conv;
import std.math;



@name("Create valid Random MDPs test")
unittest {


    foreach (i; 0 .. 50 ) {

        auto mdpStates = uniform(2, 50);
        auto mdpActions = uniform(2, 8);
        auto weight_scale = 2.0;
        auto gamma = 0.95;
        auto value_error = 0.1;

        UniqueFeaturesPerStateActionReward lr;

        auto model = generateRandomMDP(mdpStates, mdpActions, uniform(5, 20), weight_scale, gamma, lr);

        assert(model.S().size() == mdpStates, "Number of generated states is wrong");
        assert(model.A().size() == mdpActions, "Number of generated actions is wrong");

        foreach (s ; model.S()) {

            foreach (a; model.A()) {

                double sum = 0;
                
                foreach( s_p ; model.S()) {

                    auto prob = model.T()[tuple(s[0], a[0])][s_p[0]];
                    assert (prob >= 0.0, "Transitions include a negative probability");
                    assert (prob <= 1.0, "Transitions include a probability greater than 1");

                    sum += prob;                    
                    
                }

                assert(abs(sum - 1.0) < 0.000001, "Transitions don't add up to 1.0, " ~ to!string(sum));
            }
        }

        foreach (w ; lr.getWeights()) {
            assert(w <= weight_scale && w >= -weight_scale, "Feature weights incorrectly initalized");
        }
        
        auto states = model.S();
        auto actions = model.A();

        auto V = value_iteration(model, value_error / 1000 * max ( max( model.R())));
        auto policy = optimum_policy(V, model);                
        
    }

}

@name("State Visitation Frequency test")
unittest {

    foreach (iter; 0 .. 5 ) {

        auto mdpStates = uniform(2, 50);
        auto mdpActions = uniform(2, 8);
        auto weight_scale = 2.0;
        auto gamma = 0.95;
        auto value_error = 0.1;
        size_t numTrajectories = 30000;
        int trajLength = 2 * cast(int)ceil(log(value_error) / log(gamma));

        UniqueFeaturesPerStateActionReward lr;

        auto model = generateRandomMDP(mdpStates, mdpActions, uniform(5, 20), weight_scale, gamma, lr);



        auto V = value_iteration(model, value_error * max ( max( lr.toFunction())) );
        auto pi = optimum_policy(V, model);

        // find empirical state visitation frequency

        Function!(double, State) mu2 = new Function!(double, State)(model.S(), 0.0);

        foreach(i; 0 .. numTrajectories) {

            auto traj = simulate(model, to_stochastic_policy(pi, model.A()), trajLength, model.initialStateDistribution(), true );

            auto g = 1.0;
        
            foreach(timestep; traj) {

                mu2[timestep[0]] += g / numTrajectories;
                g *= gamma;
            }
        }
    
    
        auto mu1 = stateVisitationFrequency(model, pi, value_error);

        foreach (s; model.S()) {
            assert (isClose(mu1[s[0]], mu2[s[0]], 0.19), "State Visitation Frequency functions differ: 1: " ~ to!string(mu1) ~ "\n\n 2: " ~ to!string(mu2) ~ " " ~ to!string(s[0]));
        }    
        

        // test sub-rational cases

        auto V2 = soft_max_value_iteration(model, value_error * max ( max( lr.toFunction())) );
        auto pi2 = soft_max_policy(V2, model);

        // find empirical state visitation frequency

        mu2 = new Function!(double, State)(model.S(), 0.0);

        foreach(i; 0 .. numTrajectories) {

            auto traj = simulate(model, pi2, trajLength, model.initialStateDistribution(), true );

            auto g = 1.0;
        
            foreach(timestep; traj) {

                mu2[timestep[0]] += g / numTrajectories;
                g *= gamma;
            }
        }
    
    
        mu1 = stateVisitationFrequency(model, pi2, value_error);

        foreach (s; model.S()) {
            assert (isClose(mu1[s[0]], mu2[s[0]], 0.19), "State Visitation Frequency functions differ: 1: " ~ to!string(mu1) ~ "\n\n 2: " ~ to!string(mu2) ~ " " ~ to!string(s[0]));
        }    

    }
    
}

