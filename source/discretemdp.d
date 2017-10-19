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



class Space : mdp.Space {

     abstract public ulong size();
}


class StateSpace : Space {

}


class ActionSpace : Space {


}

class StateActionSpace : Space {

}

class StateActionStateSpace : Space {

}


class Distribution : mdp.Distribution {

}

class Mapping : mdp.Mapping {


}

class Model : mdp.Model {

}


class Reward {


}


class LinearReward : Reward {

}
