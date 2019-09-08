class config:
    def __init__(self, env_name):
        # env config
        self.render_train     = False
        self.render_test      = False
        self.env_name         = env_name
        self.overwrite_render = True
        self.record           = True
        self.high             = 255.

        # output config
        self.output_path  = f"results/{env_name}/"
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path + "monitor/"

        # model and training config
        self.num_episodes_test = 50
        self.grad_clip         = True
        self.clip_val          = 10
        self.saving_freq       = 250000
        self.log_freq          = 50
        self.eval_freq         = 250000
        self.record_freq       = 250000
        self.soft_epsilon      = 0.05

        # nature paper hyper params
        self.nsteps_train       = 5000000
        self.batch_size         = 32
        self.buffer_size        = 1000000
        self.target_update_freq = 10000
        self.gamma              = 0.99
        self.learning_freq      = 4
        self.state_history      = 4
        self.skip_frame         = 4
        self.lr_begin           = 0.00025
        self.lr_end             = 0.00005
        self.lr_nsteps          = self.nsteps_train/2
        self.eps_begin          = 1
        self.eps_end            = 0.1
        self.eps_nsteps         = 1000000
        self.learning_start     = 50000
