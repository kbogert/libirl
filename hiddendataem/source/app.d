
import discretemdp;
import discretefunctions;
import occlusion;
import maxentIRL;
import randommdp;
import trajectories;

import dyaml;
import std.getopt;
import std.random;
import std.stdio;
import std.conv;
import std.typecons;

enum gitcommit = import("gitcommit");
enum buildTimeStamp = __TIMESTAMP__;

void loadMDP(Node rootNode, double viTol, out Model model, out LinearReward lr) {


    int state_count = rootNode["NumStates"].as!int;
    int action_count = rootNode["NumActions"].as!int;
    double gamma = rootNode["Gamma"].as!double;
      
    RandomMDPStateSpace states = new RandomMDPStateSpace(state_count);
    RandomMDPActionSpace actions = new RandomMDPActionSpace(action_count);

    auto transitions = new ConditionalDistribution!(State, State, Action)(states, states.cartesian_product(actions));

    
    foreach(Node key, Node entry; rootNode["Transitions"]) {
        
        State temp_s = new RandomMDPState(key[0].as!int);
        Action temp_a = new RandomMDPAction(key[1].as!int);

        auto dist = new Distribution!(State)(states, 0.0);
        foreach(i; 0 .. states.size()) {
            auto temp_s2 = new RandomMDPState(cast(int)i);

            dist[temp_s2] = entry[i].as!double;
        }
        
        transitions[temp_s, temp_a] = dist;
        
    }

    auto initial = new Distribution!(State)(states, 0.0);
    foreach(i; 0 .. states.size()) {
        auto temp_s2 = new RandomMDPState(cast(int)i);

        initial[temp_s2] = rootNode["InitialStates"][i].as!double;
    }


    size_t numFeatures = rootNode["RewardFeatures"].mapping().front().value.length();

    double [] def = new double[numFeatures];
    auto featureFunc = new Function!(double [], State, Action)(states.cartesian_product(actions), def);

    foreach(Node key, Node entry; rootNode["RewardFeatures"]) {
        
        State temp_s = new RandomMDPState(key[0].as!int);
        Action temp_a = new RandomMDPAction(key[1].as!int);

        double [] mapping = new double[numFeatures];
        foreach(i; 0 .. mapping.length) {
            mapping[i] = entry[i].as!double;
        }
        
        featureFunc[temp_s, temp_a] = mapping;
        
    }    

    lr = new LinearReward(featureFunc, def);

    model = new SoftMaxModel(states, actions, transitions, lr.toFunction(), gamma, initial, viTol);
    
}

void loadTrajectories(Node rootNode, out Sequence!(State, Action)[] trajectories) {

    int i = 0;
    trajectories.length = 0;
    
    while (rootNode.containsKey("Trajectory_"~to!string(i))) {
        Sequence!(State, Action) traj = new Sequence!(State, Action)();

        foreach(Node T; rootNode["Trajectory_"~to!string(i)]) {

            State s = null;
            Action a = null;
            
            if (T[0].as!int >= 0) {
                
                s = new RandomMDPState(T[0].as!int);
            } 
            if (T[1].as!int >= 0) {
                
                a = new RandomMDPAction(T[1].as!int);
            } 

            traj ~= tuple(s, a);
        }
        
        trajectories ~= traj;
        i++;
    }
}

Node saveMDP(Model model, LinearReward lr) {

    Node numStates = Node(model.S().size());
    Node numActions = Node(model.A().size());

    Node root = Node();
    root.add("NumStates", numStates);
    root.add("NumActions", numActions);
    root.add("Gamma", model.gamma());
    
    Node T = Node();
    auto transitions = model.T();
    foreach(key; transitions.param_set()) {

        double [] transArray = new double[model.S().size()];

        foreach(s_p; model.S()) {

            transArray[ (cast(RandomMDPState)s_p[0]).getID() ] = transitions[key][s_p];
        }
        
        T.add( [(cast(RandomMDPState)key[0]).getID(), (cast(RandomMDPAction)key[1]).getID() ], transArray);
    }     
    root.add("Transitions", T);

    
    double [] initArray = new double[model.S().size()];
    foreach(s_p; model.S()) {

        initArray[ (cast(RandomMDPState)s_p[0]).getID() ] = model.initialStateDistribution()[s_p];
    }
    root.add("InitialStates", initArray);    
    

    
    Node R = Node();
    auto features = lr.getAllFeatures();
    foreach(key; features.param_set()) {
        R.add( [(cast(RandomMDPState)key[0]).getID(), (cast(RandomMDPAction)key[1]).getID() ], features[key]);
    } 
        
    root.add("RewardFeatures", R);
    return root;
    
}

Node saveTrajectories(Sequence!(State, Action)[] trajectories) {

    Node root = Node();

    foreach (t, traj; trajectories) {

        int [][] T;

        foreach(timestep; traj) {

            int stateNum;
            int actionNum;

            if (timestep[0] is null) {
                stateNum = -1;
            } else {
                stateNum = (cast(RandomMDPState)timestep[0]).getID();
            }
            
            if (timestep[1] is null) {
                actionNum = -1;
            } else {
                actionNum = (cast(RandomMDPAction)timestep[1]).getID();
            }

            T ~= [stateNum, actionNum];
        }
        
        root.add("Trajectory_"~to!string(t), T);
    }

    return root;    
}


void generateRandomMDPAndTrajectories( out Model model, out LinearReward reward, out Sequence!(State, Action)[] trajectories ) {

    UniqueFeaturesPerStateActionReward lr;
    model = generateRandomMDP(6, 3, 10, 1, 0.95, lr);
    auto occluded_states = randomOccludedStates(model, uniform(1, 5));

    // generate random trajectories with occluded timesteps
    auto arr = new Sequence!(State, Action)[10];
    foreach(j; 0 .. arr.length) {
        arr[j] = simulate(model, model.getPolicy(), uniform(15, 20), model.initialStateDistribution());
        arr[j] = removeOccludedTimesteps(arr[j], occluded_states);
    }

    reward = lr;
    trajectories = arr;

        
}

void main(string[] args) {


    // read in parameters:
    // mdp file
    // trajectories file
    // settings: Value error, gradientDescent tolerance, useStochasticGD, LME Tolerance, LME max iterations

    string mdpFile;
    string trajFile;
    double valueError = 0.1;
    bool useStochasticGD = true;
    double gdTolerance = 0.001;
    double emTolerance = 0.1;
    long emIterations = 100;
    bool debugOn = false;
    bool writeOut = false;
    bool strictMode = false;

    long burnInSamples = 1000;
    long totalMCMCSamples = 10000;
    size_t MCMCRepeats = 10;
    enum MStepType { exact, approx, empiricalApprox }
    enum EStepType { exact, gibbs, hybrid }
    MStepType mStep = MStepType.exact;
    EStepType eStep = EStepType.exact;
    
    auto helpInformation = getopt(
    args,
    std.getopt.config.required,
    "mdp", "YAML file descibing the incomplete MDP", &mdpFile,
    std.getopt.config.required,
    "traj", "TAML File descibing the observed trajectories", &trajFile,
    "valueError",  "Relative value error to stop value iteration", &valueError,    
    "stochasticGD", "Use stochastic gradient descent if true, else gradient descent (Unconstrained Adaptive Gradient Descent, Steinhardt & Liang 2014)", &useStochasticGD,
    "GDTolerance", "Gradient Descent tolerance, ask Kenny", &gdTolerance,
    "EMTolerance", "Expectation Maximization tolerance, see HiddenDataEM paper", &emTolerance,
    "EMMaxIterations", "Expectation Maximization maximum iterations, see HiddenDataEM paper", &emIterations,
    "debug", "Turn debug output on", &debugOn,
    "write", "Generate a random MDP and write the YAML files from it", &writeOut,
    "mstep", "Type of M Step to use (exact (default), approx, empiricalApprox)", &mStep,
    "estep", "Type of E Step to use (exact (default), gibbs, hybrid)", &eStep,
    "burninsamples", "Number of samples for MCMC burn in", &burnInSamples,
    "totalsamples", "Total Number of samples for MCMC (Must be > burn in)", &totalMCMCSamples,
    "repeats", "MCMC repeat starts", &MCMCRepeats        
    );

    if (helpInformation.helpWanted)
    {
    defaultGetoptPrinter("Run Hidden Data EM on the specified MDP and trajectory files (all assumed to be YAML).\n\nBuild date: " ~buildTimeStamp ~ " from commit " ~ gitcommit,
      helpInformation.options);
      return;
    }

    
    if (writeOut) {

        Model model;
        LinearReward reward;
        Sequence!(State, Action)[] trajectories;

        generateRandomMDPAndTrajectories(model, reward, trajectories);

        Node root = saveMDP(model, reward);        
        dumper.dump(File(mdpFile, "w").lockingTextWriter, root);

        root = saveTrajectories(trajectories);        
        dumper.dump(File(trajFile, "w").lockingTextWriter, root);

        return;
    }

    Model model;
    LinearReward reward;
    Sequence!(State, Action)[] trajectories;

    Node root = Loader.fromFile(mdpFile).load();
    loadMDP(root, valueError, model, reward);

    root = Loader.fromFile(trajFile).load();
    loadTrajectories(root, trajectories);


    Sequence_MaxEnt_Problem!(State, Action) M_step; 
    Sequence_Distribution_Computer!(State, Action) E_step; 

    final switch(mStep) {
    case MStepType.exact:
        M_step = new MaxCausalEntIRL_InfMDP(model, reward, gdTolerance, reward.getWeights(), useStochasticGD, debugOn);
        break;
    case MStepType.approx:
        M_step = new MaxCausalEntIRL_SGDApprox(model, reward, gdTolerance, reward.getWeights(), useStochasticGD, debugOn);
        break;
    case MStepType.empiricalApprox:
        M_step = new MaxCausalEntIRL_SGDEmpirical(model, reward, gdTolerance, reward.getWeights(), useStochasticGD, debugOn);
        break;
    }

    final switch(eStep) {
    case EStepType.exact:
        E_step = new ExactPartialTrajectoryToTrajectoryDistr(model, reward);
        break;
    case EStepType.gibbs:
        E_step = new GibbsSamplingApproximatePartialTrajectoryToTrajectoryDistr(model, reward, MCMCRepeats, burnInSamples, totalMCMCSamples, null, true, debugOn);
        break;
    case EStepType.hybrid:

        // build a uniform proposal distribution

        Sequence!(Distribution!(State, Action))[] uniformProposal;
        auto u = new Distribution!(State, Action)(model.S().cartesian_product(model.A()), DistInitType.Uniform);
        Sequence!(Distribution!(State, Action)) [] uniform_sa_sequence;
        uniformProposal.length = trajectories.length;

        foreach(i; 0 .. trajectories.length) {
            uniformProposal[i] = new Sequence!(Distribution!(State, Action))(trajectories[i].length());
            foreach(t; 0 .. trajectories[i].length()) {
                uniformProposal[i][t] = tuple(u);
            }
        }
        
        E_step = new HybridMCMCApproximatePartialTrajectoryToTrajectoryDistr(model, reward, MCMCRepeats, burnInSamples, totalMCMCSamples, uniformProposal, null, true, debugOn);
        break;
    }

    if (debugOn) {
        writeln("Config loaded, beginning solver");
    }
    auto lme_irl = new LME_IRL!(State, Action)(M_step, E_step, emTolerance, emIterations, debugOn);

    double [] rand_weights = new double[reward.getSize()];
    foreach(ref w; rand_weights) {
        w = uniform(0.001, 0.05);
    }
    auto found_weights = lme_irl.solve(trajectories, rand_weights);

    writeln(found_weights);
        
}

