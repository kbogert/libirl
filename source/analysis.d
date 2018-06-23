module analysis;

import discretemdp;
import discretefunctions;

double calcInverseLearningError(Model model, Function!(double, State) V_pi_star, Function!(double, State) V_pi_L) {

    auto diff = V_pi_star - V_pi_L;

    return abs(diff).sumout();

}

double calcInverseLearningError(Model model, Reward true_reward, Reward learned_reward, double tolerance, int max_iter = int.max) {

    auto true_model = new BasicModel(model.S(), model.A(), model.T(), true_reward.toFunction(), model.gamma(), model.initialStateDistribution());
    auto V_pi_star = value_iteration(true_model, tolerance, max_iter);

    auto temp_model = new BasicModel(model.S(), model.A(), model.T(), learned_reward.toFunction(), model.gamma(), model.initialStateDistribution());
    auto learned_value = value_iteration(temp_model, tolerance, max_iter);
    auto pi_L = optimum_policy(learned_value, true_model);

    auto V_pi_L = value_function_under_policy(true_model, pi_L, tolerance, max_iter);

    return calcInverseLearningError(true_model, V_pi_star, V_pi_L);
}
