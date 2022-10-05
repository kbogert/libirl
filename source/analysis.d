module analysis;

import discretemdp;
import discretefunctions;
import std.typecons;

double calcInverseLearningError(Function!(double, State) V_pi_star, Function!(double, State) V_pi_L) {

    auto diff = V_pi_star - V_pi_L;

    return abs(diff).sumout();

}

double calcInverseLearningError(Model model, Reward true_reward, Reward learned_reward, double tolerance, int max_iter = int.max) {

    auto temp_rewards = model.R();

    try {
        
        model.setR(true_reward.toFunction());
        auto Q = q_value_iteration(model, tolerance * max ( max( abs(true_reward.toFunction()))), max_iter);
        auto pi_star = optimum_policy(Q);
    
        model.setR(learned_reward.toFunction());
        auto learned_value = value_iteration(model, tolerance * max ( max( abs(learned_reward.toFunction()))), max_iter);
        auto pi_L = optimum_policy(learned_value, model);

        auto V_pi_star = Q.apply(pi_star);
        auto V_pi_L = Q.apply(pi_L);

        return calcInverseLearningError(V_pi_star, V_pi_L);

    } finally {
        model.setR(temp_rewards);
    }
    
}

double learnedBehaviorAccuracy(Model model, Function!(Tuple!(Action), State) true_policy, Function!(Tuple!(Action), State) learned_policy) {

    size_t correct_count = 0;

    foreach (s; model.S()) {

        if (true_policy[s[0]] == learned_policy[s[0]])
            correct_count ++;
    }

    return correct_count / cast(double) model.S().size();

}


double calcStochasticInverseLearningError(Model model, Reward true_reward, Reward learned_reward, double tolerance, int max_iter = int.max) {
    auto temp_rewards = model.R();

    try {
        
        model.setR(true_reward.toFunction());
        auto pi_star = model.getPolicy();
    
        model.setR(learned_reward.toFunction());
        auto pi_L = model.getPolicy();

        auto mu_pi_star = stateActionVisitationFrequency(model, pi_star, tolerance, max_iter);
        auto mu_pi_L = stateActionVisitationFrequency(model, pi_L, tolerance, max_iter);

        auto V_pi_star = sumout!(Action)(mu_pi_star * true_reward.toFunction());
        auto V_pi_L = sumout!(Action)(mu_pi_L * true_reward.toFunction());
        
        return calcInverseLearningError(V_pi_star, V_pi_L);

    } finally {
        model.setR(temp_rewards);
    }

}
