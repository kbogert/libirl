module singletest;

import tested;
import randommdp;
import discretemdp;
import discretefunctions;
import occlusion;
import maxentIRL;
import analysis;
import gridworld;
import trajectories;
import solvers;
import featureexpectations;
import partialobservation;

import std.stdio;
import std.typecons;
import std.conv;
import std.random;
import std.math;


@name("State Visitation Frequency test")
unittest {

    foreach (iter; 0 .. 5 ) {

        auto mdpStates = uniform(2, 50);
        auto mdpActions = uniform(2, 8);
        auto weight_scale = 2.0;
        auto gamma = 0.95;
        auto value_error = 0.01;
        size_t numTrajectories = 2000*mdpStates*mdpActions;
        int trajLength = cast(int)ceil(log(value_error / 2) / log(gamma));

        UniqueFeaturesPerStateActionReward lr;

        auto model = generateRandomMDP(mdpStates, mdpActions, uniform(5, 20), weight_scale, gamma, lr);



        auto V = value_iteration(model, value_error * max ( max( lr.toFunction())) );
        auto pi = optimum_policy(V, model);

        // find empirical state visitation frequency

        Function!(double, State) mu2 = new Function!(double, State)(model.S(), 0.0);

        auto stoc_policy = to_stochastic_policy(pi, model.A());
        foreach(i; 0 .. numTrajectories) {

            auto traj = simulate(model, stoc_policy, trajLength, model.initialStateDistribution(), true );

            auto g = 1.0;
        
            foreach(timestep; traj) {

                mu2[timestep[0]] += g / numTrajectories;
                g *= gamma;
            }
        }
    
    
        auto mu1 = stateVisitationFrequency(model, pi, value_error);

        foreach (s; model.S()) {
            writeln(abs(mu1[s[0]] - mu2[s[0]]), " : ", 0.01 * mu1[s[0]], " ", mu1[s[0]]);
            assert (isClose(mu1[s[0]], mu2[s[0]], 0.01, 1e-5), "State Visitation Frequency functions differ: 1: " ~ to!string(mu1) ~ "\n\n 2: " ~ to!string(mu2) ~ " " ~ to!string(s[0]));
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
            assert (isClose(mu1[s[0]], mu2[s[0]], 0.01, 1e-5), "State Visitation Frequency functions differ: 1: " ~ to!string(mu1) ~ "\n\n 2: " ~ to!string(mu2) ~ " " ~ to!string(s[0]));
        }    

    }
    
}
