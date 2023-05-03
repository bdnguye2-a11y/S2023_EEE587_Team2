from SampleLinearModel import LinearSystem


cart_pos_x=4

cart_pos_y=1

class PID:
    def __init__(self, kp, ki, kd):
        
        self.plant=LinearSystem()
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.last_error = 0
        self.integral = 0
        self.error=0
        self.Y_ref=0

    def update(self,  dt):

        self.error=self.Y_ref - self.plant.Y[1]
        self.integral += self.error * dt
        derivative = (self.error - self.last_error) / dt
        self.plant.U = self.kp * self.error + self.ki * self.integral + self.kd * derivative
        self.last_error = self.error
        
        return
    
    def do_PID(self, ref):
        self.Y_ref=ref
        dt=0.01
        sim_time=15 # in seconds
        N = sim_time/(dt)

        self.plant.U =1 # initial U
        for iter in range(int(N)):

            # Update plant
            self.plant.State_dot=self.plant.A@self.plant.State +self.plant.B*self.plant.U
            self.plant.Y = self.plant.C@self.plant.State
            self.plant.State = self.plant.State + self.plant.State_dot*dt

            print(self.plant.Y[1])
            if iter %10==0:
                self.plant.animate_system()
            #Update control input based on PID 
            self.update(dt)
        

controller = PID(10, 1, 20)
controller.do_PID(0)

        
