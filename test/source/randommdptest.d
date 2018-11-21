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
