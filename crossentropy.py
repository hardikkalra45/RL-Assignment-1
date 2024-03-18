class CrossEntropy:

    def __init__(self, environment, test_environment, num_episodes=30) -> None:
        self.environment = environment
        self.test_env = test_environment
        self.num_episodes = num_episodes
        self.K = 10
        self.K_epsilon = 5
        self.epsilon = 0.02
        self.environment.reset()
        self.state_space = 5
        self.action_space = 2
        self.params = np.random.rand(self.state_space)
        self.gains=[]
        self.sigma = 2 * np.eye(self.state_space)

    def update_sigma(self):
        theta_ks = [np.random.multivariate_normal(self.params, self.sigma) for i in range(self.K)]
        self.test_env.reset()
        theta_gains = sorted([(self.calculate_gains(self.test_env, theta_k),theta_k) for theta_k in theta_ks], key= lambda x: x[0],
                             reverse= True)
        top_thetas = [b for i,(a,b) in enumerate(theta_gains) if i < self.K_epsilon]
        theta_mean = np.mean(top_thetas, axis=0)
        new_sigma = (1/(self.epsilon + self.K_epsilon)) * ( ((2*self.epsilon)*np.eye(5))
                                                           + sum([(theta - theta_mean) @ (theta - theta_mean).T
                                                                  for theta in top_thetas]) )
        return new_sigma

    def update_params(self, params, sigma, current_reward):

        self.test_env.reset()
        while True:
            new_params = np.random.multivariate_normal(params, sigma)
            new_reward = self.calculate_gains(self.test_env , new_params)
            improvement = new_reward - current_reward
            if abs(improvement) < self.epsilon or current_reward == 1001:
                return params, True, current_reward
            elif improvement > 0:
                return new_params, False, new_reward
            else:
                pass


    def parameterized_gradient_ascent(self, suppress_output = False):

        converged = False
        for episode in range(self.num_episodes):
            state = self.environment.reset()
            reward = self.calculate_gains(self.environment,self.params)

            if not suppress_output:
                print(f"Episode No. {episode + 1}, Total Reward Obtained: {reward}, Converged = {converged}")
            self.gains.append(reward)

            if not converged:
                self.params, converged, _ = self.update_params(self.params, self.sigma, reward)
                self.sigma = self.update_sigma()

        return self.params

    def calculate_gains(self, env, test_params):
        state = env.reset()
        total_reward =0
        while True:
            action = policy(state, test_params)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            state = next_state

            if done:
                break

        return total_reward

    def reset(self):
        self.environment.reset()
        self.test_env.reset()
        self.params = np.random.rand(self.state_space)
        self.gains=[]
        self.sigma = 2 * np.eye(self.state_space)
