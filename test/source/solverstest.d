module solverstest;

import solvers;
import tested;
import discretefunctions;
import std.math;
import std.typecons;
import std.conv;


@name("Markov Smoother test")
unittest {

    // Using the umbrella example from Russell and Norvig
    auto states = new NumericSetSpace(1, 3);

    double [Tuple!(size_t)][Tuple!(size_t)] init_states;
    init_states[tuple(1UL)][tuple(1UL)] = 0.7;
    init_states[tuple(1UL)][tuple(2UL)] = 0.3;
    init_states[tuple(2UL)][tuple(1UL)] = 0.3;
    init_states[tuple(2UL)][tuple(2UL)] = 0.7;
    
    auto transitions = new ConditionalDistribution!(size_t, size_t)(states, states, init_states) ;   


    auto observations = new Sequence!(Distribution!(size_t))(5);

    double [Tuple!(size_t)] obs0;
    obs0[tuple(1UL)] = 0.9;
    obs0[tuple(2UL)] = 0.2;
    
    observations[0] = tuple(new Distribution!(size_t)(states, obs0));

    double [Tuple!(size_t)] obs1;
    obs1[tuple(1UL)] = 0.9;
    obs1[tuple(2UL)] = 0.2;
    
    observations[1] = tuple(new Distribution!(size_t)(states, obs1));

    double [Tuple!(size_t)] obs2;
    obs2[tuple(1UL)] = 0.1;
    obs2[tuple(2UL)] = 0.8;
    
    observations[2] = tuple(new Distribution!(size_t)(states, obs2));
    
    double [Tuple!(size_t)] obs3;
    obs3[tuple(1UL)] = 0.9;
    obs3[tuple(2UL)] = 0.2;
    
    observations[3] = tuple(new Distribution!(size_t)(states, obs3));

    double [Tuple!(size_t)] obs4;
    obs4[tuple(1UL)] = 0.9;
    obs4[tuple(2UL)] = 0.2;
    
    observations[4] = tuple(new Distribution!(size_t)(states, obs4));


    auto initial = new Distribution!(size_t)(states, DistInitType.Uniform);
    
    auto results = SequenceMarkovChainSmoother!(size_t)(observations, transitions, initial);

    assert(approxEqual(results[0][0][tuple(1UL)], 0.8637), "Markov Smoother 1");
    assert(approxEqual(results[1][0][tuple(1UL)], 0.8204), "Markov Smoother 2");
    assert(approxEqual(results[2][0][tuple(1UL)], 0.3075), "Markov Smoother 3");    
    assert(approxEqual(results[3][0][tuple(1UL)], 0.8204), "Markov Smoother 4");
    assert(approxEqual(results[4][0][tuple(1UL)], 0.8637), "Markov Smoother 5");

    assert(results[0][0].isNormalized(), "Not Normalized");
    assert(results[1][0].isNormalized(), "Not Normalized");
    assert(results[2][0].isNormalized(), "Not Normalized");
    assert(results[3][0].isNormalized(), "Not Normalized");
    assert(results[4][0].isNormalized(), "Not Normalized");
     
}



@name("Markov Gibbs Sampler test")
unittest {

    // Using the umbrella example from Russell and Norvig
    auto states = new NumericSetSpace(1, 3);

    double [Tuple!(size_t)][Tuple!(size_t)] init_states;
    init_states[tuple(1UL)][tuple(1UL)] = 0.7;
    init_states[tuple(1UL)][tuple(2UL)] = 0.3;
    init_states[tuple(2UL)][tuple(1UL)] = 0.3;
    init_states[tuple(2UL)][tuple(2UL)] = 0.7;
    
    auto transitions = new ConditionalDistribution!(size_t, size_t)(states, states, init_states) ;   


    auto observations = new Sequence!(Distribution!(size_t))(5);

    double [Tuple!(size_t)] obs0;
    obs0[tuple(1UL)] = 0.9;
    obs0[tuple(2UL)] = 0.2;
    
    observations[0] = tuple(new Distribution!(size_t)(states, obs0));

    double [Tuple!(size_t)] obs1;
    obs1[tuple(1UL)] = 0.9;
    obs1[tuple(2UL)] = 0.2;
    
    observations[1] = tuple(new Distribution!(size_t)(states, obs1));

    double [Tuple!(size_t)] obs2;
    obs2[tuple(1UL)] = 0.1;
    obs2[tuple(2UL)] = 0.8;
    
    observations[2] = tuple(new Distribution!(size_t)(states, obs2));
    
    double [Tuple!(size_t)] obs3;
    obs3[tuple(1UL)] = 0.9;
    obs3[tuple(2UL)] = 0.2;
    
    observations[3] = tuple(new Distribution!(size_t)(states, obs3));

    double [Tuple!(size_t)] obs4;
    obs4[tuple(1UL)] = 0.9;
    obs4[tuple(2UL)] = 0.2;
    
    observations[4] = tuple(new Distribution!(size_t)(states, obs4));


    auto initial = new Distribution!(size_t)(states, DistInitType.Uniform);

    foreach (i; 0 .. 10) {
        auto results = MarkovGibbsSampler!(size_t)(observations, transitions, initial, 10000, 1500000);

        assert(approxEqual(results[0][0][tuple(1UL)], 0.8637), "Markov Smoother 1 " ~ to!string(results[0][0][tuple(1UL)]));
        assert(approxEqual(results[1][0][tuple(1UL)], 0.8204), "Markov Smoother 2 " ~ to!string(results[1][0][tuple(1UL)]));
        assert(approxEqual(results[2][0][tuple(1UL)], 0.3075), "Markov Smoother 3 " ~ to!string(results[2][0][tuple(1UL)]));    
        assert(approxEqual(results[3][0][tuple(1UL)], 0.8204), "Markov Smoother 4 " ~ to!string(results[3][0][tuple(1UL)]));
        assert(approxEqual(results[4][0][tuple(1UL)], 0.8637), "Markov Smoother 5 " ~ to!string(results[4][0][tuple(1UL)]));

        assert(results[0][0].isNormalized(), "Not Normalized");
        assert(results[1][0].isNormalized(), "Not Normalized");
        assert(results[2][0].isNormalized(), "Not Normalized");
        assert(results[3][0].isNormalized(), "Not Normalized");
        assert(results[4][0].isNormalized(), "Not Normalized");
    }     
}


@name("Hybrid MCMC Sampler test")
unittest {

    // Using the umbrella example from Russell and Norvig
    auto states = new NumericSetSpace(1, 3);

    double [Tuple!(size_t)][Tuple!(size_t)] init_states;
    init_states[tuple(1UL)][tuple(1UL)] = 0.7;
    init_states[tuple(1UL)][tuple(2UL)] = 0.3;
    init_states[tuple(2UL)][tuple(1UL)] = 0.3;
    init_states[tuple(2UL)][tuple(2UL)] = 0.7;
    
    auto transitions = new ConditionalDistribution!(size_t, size_t)(states, states, init_states) ;   


    auto observations = new Sequence!(Distribution!(size_t))(5);

    double [Tuple!(size_t)] obs0;
    obs0[tuple(1UL)] = 0.9;
    obs0[tuple(2UL)] = 0.2;
    
    observations[0] = tuple(new Distribution!(size_t)(states, obs0));

    double [Tuple!(size_t)] obs1;
    obs1[tuple(1UL)] = 0.9;
    obs1[tuple(2UL)] = 0.2;
    
    observations[1] = tuple(new Distribution!(size_t)(states, obs1));

    double [Tuple!(size_t)] obs2;
    obs2[tuple(1UL)] = 0.1;
    obs2[tuple(2UL)] = 0.8;
    
    observations[2] = tuple(new Distribution!(size_t)(states, obs2));
    
    double [Tuple!(size_t)] obs3;
    obs3[tuple(1UL)] = 0.9;
    obs3[tuple(2UL)] = 0.2;
    
    observations[3] = tuple(new Distribution!(size_t)(states, obs3));

    double [Tuple!(size_t)] obs4;
    obs4[tuple(1UL)] = 0.9;
    obs4[tuple(2UL)] = 0.2;
    
    observations[4] = tuple(new Distribution!(size_t)(states, obs4));


    auto initial = new Distribution!(size_t)(states, DistInitType.Uniform);

    foreach (i; 0 .. 10) {
        auto results = HybridMCMC!(size_t)(observations, transitions, initial, observations, 10000, 2500000);

        assert(approxEqual(results[0][0][tuple(1UL)], 0.8637), "Markov Smoother 1 " ~ to!string(results[0][0][tuple(1UL)]));
        assert(approxEqual(results[1][0][tuple(1UL)], 0.8204), "Markov Smoother 2 " ~ to!string(results[1][0][tuple(1UL)]));
        assert(approxEqual(results[2][0][tuple(1UL)], 0.3075), "Markov Smoother 3 " ~ to!string(results[2][0][tuple(1UL)]));    
        assert(approxEqual(results[3][0][tuple(1UL)], 0.8204), "Markov Smoother 4 " ~ to!string(results[3][0][tuple(1UL)]));
        assert(approxEqual(results[4][0][tuple(1UL)], 0.8637), "Markov Smoother 5 " ~ to!string(results[4][0][tuple(1UL)]));

        assert(results[0][0].isNormalized(), "Not Normalized");
        assert(results[1][0].isNormalized(), "Not Normalized");
        assert(results[2][0].isNormalized(), "Not Normalized");
        assert(results[3][0].isNormalized(), "Not Normalized");
        assert(results[4][0].isNormalized(), "Not Normalized");
    }     
}


@name("Adaptive Hybrid MCMC Sampler test")
unittest {

    // Using the umbrella example from Russell and Norvig
    auto states = new NumericSetSpace(1, 3);

    double [Tuple!(size_t)][Tuple!(size_t)] init_states;
    init_states[tuple(1UL)][tuple(1UL)] = 0.7;
    init_states[tuple(1UL)][tuple(2UL)] = 0.3;
    init_states[tuple(2UL)][tuple(1UL)] = 0.3;
    init_states[tuple(2UL)][tuple(2UL)] = 0.7;
    
    auto transitions = new ConditionalDistribution!(size_t, size_t)(states, states, init_states) ;   


    auto observations = new Sequence!(Distribution!(size_t))(5);

    double [Tuple!(size_t)] obs0;
    obs0[tuple(1UL)] = 0.9;
    obs0[tuple(2UL)] = 0.2;
    
    observations[0] = tuple(new Distribution!(size_t)(states, obs0));

    double [Tuple!(size_t)] obs1;
    obs1[tuple(1UL)] = 0.9;
    obs1[tuple(2UL)] = 0.2;
    
    observations[1] = tuple(new Distribution!(size_t)(states, obs1));

    double [Tuple!(size_t)] obs2;
    obs2[tuple(1UL)] = 0.1;
    obs2[tuple(2UL)] = 0.8;
    
    observations[2] = tuple(new Distribution!(size_t)(states, obs2));
    
    double [Tuple!(size_t)] obs3;
    obs3[tuple(1UL)] = 0.9;
    obs3[tuple(2UL)] = 0.2;
    
    observations[3] = tuple(new Distribution!(size_t)(states, obs3));

    double [Tuple!(size_t)] obs4;
    obs4[tuple(1UL)] = 0.9;
    obs4[tuple(2UL)] = 0.2;
    
    observations[4] = tuple(new Distribution!(size_t)(states, obs4));


    auto initial = new Distribution!(size_t)(states, DistInitType.Uniform);

    foreach (i; 0 .. 10) {
        auto results = AdaptiveHybridMCMC!(size_t)(observations, transitions, initial, observations, 10000, 2000000);

        assert(approxEqual(results[0][0][tuple(1UL)], 0.8637), "Markov Smoother 1 " ~ to!string(results[0][0][tuple(1UL)]));
        assert(approxEqual(results[1][0][tuple(1UL)], 0.8204), "Markov Smoother 2 " ~ to!string(results[1][0][tuple(1UL)]));
        assert(approxEqual(results[2][0][tuple(1UL)], 0.3075), "Markov Smoother 3 " ~ to!string(results[2][0][tuple(1UL)]));    
        assert(approxEqual(results[3][0][tuple(1UL)], 0.8204), "Markov Smoother 4 " ~ to!string(results[3][0][tuple(1UL)]));
        assert(approxEqual(results[4][0][tuple(1UL)], 0.8637), "Markov Smoother 5 " ~ to!string(results[4][0][tuple(1UL)]));

        assert(results[0][0].isNormalized(), "Not Normalized");
        assert(results[1][0].isNormalized(), "Not Normalized");
        assert(results[2][0].isNormalized(), "Not Normalized");
        assert(results[3][0].isNormalized(), "Not Normalized");
        assert(results[4][0].isNormalized(), "Not Normalized");
    }     
}



@name("Adaptive Hybrid MCMC Importance Sampler test")
unittest {

    // Using the umbrella example from Russell and Norvig
    auto states = new NumericSetSpace(1, 3);

    double [Tuple!(size_t)][Tuple!(size_t)] init_states;
    init_states[tuple(1UL)][tuple(1UL)] = 0.7;
    init_states[tuple(1UL)][tuple(2UL)] = 0.3;
    init_states[tuple(2UL)][tuple(1UL)] = 0.3;
    init_states[tuple(2UL)][tuple(2UL)] = 0.7;
    
    auto transitions = new ConditionalDistribution!(size_t, size_t)(states, states, init_states) ;   


    auto observations = new Sequence!(Distribution!(size_t))(5);

    double [Tuple!(size_t)] obs0;
    obs0[tuple(1UL)] = 0.9;
    obs0[tuple(2UL)] = 0.2;
    
    observations[0] = tuple(new Distribution!(size_t)(states, obs0));

    double [Tuple!(size_t)] obs1;
    obs1[tuple(1UL)] = 0.9;
    obs1[tuple(2UL)] = 0.2;
    
    observations[1] = tuple(new Distribution!(size_t)(states, obs1));

    double [Tuple!(size_t)] obs2;
    obs2[tuple(1UL)] = 0.1;
    obs2[tuple(2UL)] = 0.8;
    
    observations[2] = tuple(new Distribution!(size_t)(states, obs2));
    
    double [Tuple!(size_t)] obs3;
    obs3[tuple(1UL)] = 0.9;
    obs3[tuple(2UL)] = 0.2;
    
    observations[3] = tuple(new Distribution!(size_t)(states, obs3));

    double [Tuple!(size_t)] obs4;
    obs4[tuple(1UL)] = 0.9;
    obs4[tuple(2UL)] = 0.2;
    
    observations[4] = tuple(new Distribution!(size_t)(states, obs4));


    auto initial = new Distribution!(size_t)(states, DistInitType.Uniform);

    foreach (i; 0 .. 10) {
        auto results = AdaptiveHybridMCMCIS!(size_t)(observations, transitions, initial, observations, 10000, 1500000);

        assert(approxEqual(results[0][0][tuple(1UL)], 0.8637), "Markov Smoother 1 " ~ to!string(results[0][0][tuple(1UL)]));
        assert(approxEqual(results[1][0][tuple(1UL)], 0.8204), "Markov Smoother 2 " ~ to!string(results[1][0][tuple(1UL)]));
        assert(approxEqual(results[2][0][tuple(1UL)], 0.3075), "Markov Smoother 3 " ~ to!string(results[2][0][tuple(1UL)]));    
        assert(approxEqual(results[3][0][tuple(1UL)], 0.8204), "Markov Smoother 4 " ~ to!string(results[3][0][tuple(1UL)]));
        assert(approxEqual(results[4][0][tuple(1UL)], 0.8637), "Markov Smoother 5 " ~ to!string(results[4][0][tuple(1UL)]));

        assert(results[0][0].isNormalized(), "Not Normalized");
        assert(results[1][0].isNormalized(), "Not Normalized");
        assert(results[2][0].isNormalized(), "Not Normalized");
        assert(results[3][0].isNormalized(), "Not Normalized");
        assert(results[4][0].isNormalized(), "Not Normalized");
    }     
}


