#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include <vector>
#include <random>
#include <numeric> 
#include <iostream>
//Calibrate and code rotational frames

//particle prediction is done by weighted avg of particles

//TODO: understand yaw and pitch formulations from theta!!

//TODO thetaInflight declaration;
    //angles from reference 

using namespace Eigen;

struct Particle{
    Eigen::Vector3f position;
    Eigen::Vector3f velocity;      // Orientation (qw, qx, qy, qz)
    Eigen::Vector3f accel;
    Eigen::Vector3f theta;  // Acceleration (ax, ay, az)
    Eigen::Vector3f omega;        // Angular velocity (wx, wy, wz)
    Eigen::Vector3f alpha;  

    Particle()
    : position(Eigen::Vector3f::Zero()),
      velocity(Eigen::Vector3f::Zero()),
      accel(Eigen::Vector3f::Zero()),
      theta(Eigen::Vector3f::Zero()),
      omega(Eigen::Vector3f::Zero()),
      alpha(Eigen::Vector3f::Zero()) {}
    
    Particle(const Eigen::Vector3f& p,
        const Eigen::Vector3f& v, 
        const Eigen::Vector3f& a, 
        const Eigen::Vector3f& t, 
        const Eigen::Vector3f& w, 
        const Eigen::Vector3f& al)
   : position(p), velocity(v), accel(a),theta(t), omega(w), alpha(al) {}
};// Magnetometer (bx, by, bz)     // Position (x, y, z)    // Velocity (vx, vy, vz)  


struct RandomizedComponents {
    Eigen::Vector3f rand_position;
    Eigen::Vector3f rand_acceleration;
    Eigen::Vector3f rand_angular_velocity;  // Consider better naming based on your use case
    
    // Required for Eigen memory alignment
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct StateVector {
    Eigen::Vector3f position;
    Eigen::Vector3f acceleration;
    Eigen::Vector3f omega;
    Eigen::Vector3f magnet;
};

class ParticleFilter {
    public:
        static const int N=200;
        std::vector<Particle> particles;
        StateVector currState; 
        Particle StateParticle;

        Eigen::Vector3f currState_position;
        Eigen::Vector3f currState_acceleration;
        Eigen::Vector3f currState_omega;
        Eigen::Vector3f currState_magnet;

        static float varianceAccel;
        static float varianceMagnet;
        static float variancePosition;
        static float varianceGyro;
        static float meanAccel;
        static float meanMagnet;
        static float meanPosition;
        static float meanGyro;

        Eigen::Quaternionf q_normalized = Eigen::Quaternionf::Identity();

        //first four are quaternion
        //next 3 are velocity
        //next three are acceleration
        //next three are thetas
        //next three are omegas
        //next three are alphas 

        float particleWeights[N];
        float refresh_rate = .1;

        bool setup(float meanAccel, float varianceAccel, float meanMagnet, float varianceMagnet, float meanPosition,float variancePosition,float meanGyro,float varianceGyro,const StateVector& currState);
        void InitializeParticles(const StateVector& currState);
        void ProgressParticles_prelaunch(const StateVector& currState);
        void ProgressParticles_postlaunch(const StateVector& currState);
        void resampleParticles();
        void reweightParticles();
        void filterParticles();
        RandomizedComponents randomize();
        Particle returnParticle();
    
};


//debug code
float ParticleFilter::varianceAccel = 0.0f;
float ParticleFilter::varianceMagnet = 0.0f;
float ParticleFilter::variancePosition = 0.0f;
float ParticleFilter::varianceGyro = 0.0f;
float ParticleFilter::meanAccel = 0.0f;
float ParticleFilter::meanMagnet = 0.0f;
float ParticleFilter::meanPosition = 0.0f;
float ParticleFilter::meanGyro = 0.0f;

// For static const int N (if needed for address-taking):
const int ParticleFilter::N;


    bool ParticleFilter::setup(float meanAccel, float varianceAccel, float meanMagnet, float varianceMagnet, float meanPosition,float variancePosition,float meanGyro,float varianceGyro,const StateVector& currState){
        ParticleFilter::varianceAccel = varianceAccel;
        ParticleFilter::varianceMagnet = varianceMagnet;
        ParticleFilter::variancePosition = variancePosition;
        ParticleFilter::varianceGyro = varianceGyro;
        ParticleFilter::meanPosition = meanPosition; 
        ParticleFilter::meanAccel = meanAccel;
        ParticleFilter::meanMagnet = meanMagnet;
        ParticleFilter::meanPosition = meanPosition;
        ParticleFilter::meanGyro = meanGyro;
        //currState = <vector of vectors>
        ParticleFilter::currState_position =currState.position;
        ParticleFilter::currState_acceleration=currState.acceleration;
        ParticleFilter::currState_omega = currState.omega;
        ParticleFilter::currState_magnet = currState.magnet;

        return true;
        

    }

    RandomizedComponents ParticleFilter::randomize(){

        //Mersenne Twiester pesudo-random number generator
        //after 2^19937 - 1 iterations, it will cycle
        //Its getting reset for each call from epoch time.
        std::random_device rd; 
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> accel_pdf(0,ParticleFilter::varianceAccel);
        
        std::uniform_real_distribution<float> position_pdf(0,ParticleFilter::variancePosition);
        
        std::uniform_real_distribution<float> gyro_pdf(0,ParticleFilter::varianceGyro);

        std::uniform_real_distribution<float> magnet_pdf(0,ParticleFilter::varianceMagnet);

            Eigen::Vector3f rand_accelVector={accel_pdf(gen),accel_pdf(gen),accel_pdf(gen)};
            Eigen::Vector3f rand_positionVector={position_pdf(gen),position_pdf(gen),position_pdf(gen)};
            Eigen::Vector3f rand_thetaVector= {gyro_pdf(gen),gyro_pdf(gen),magnet_pdf(gen)};

            //weird stuff here, commented version was working logic with my local particle filter
            // return {
            //     .rand_position = rand_positionVector,
            //     .rand_acceleration = rand_accelVector,
            //     .rand_angular_velocity = rand_thetaVector
            // };


            return {
                rand_positionVector,
                rand_accelVector,
                rand_thetaVector
            };
        }
    
        //what is this randomize function, to take in a vector of particles and tale random vectros each with some normal gaussian distribution

    void ParticleFilter::InitializeParticles(const StateVector& currState){
    //TODO change weight to a vectorized update.
        //what is the order of computation? 
            //sensor fuse angles then initialize quaternion, then initialize particles with sensor vals in world frame!
        
            //theta = a(theta + wdt) + (1-a)theta_2
        //x_rot = arctan2(-ay/cos(theta_x),-az/cos(theta_x))
        //x_arcsin(a_x/g)

        //!!computing angular values by sensor fusion!!
        float a = 0.4;
        float theta_x1=0.0;
        float theta_x2=0.0;
        float theta_y1=0.0;
        float theta_y2=0.0;
        float theta_z1=0.0;
        float theta_z2=0.0;

        float theta_xfused=0.0;
        float theta_yfused=0.0;
        float theta_zfused=0.0;

        //TODO incorporate divide by 0 error logic by some 1e-6

        theta_x1 = currState.omega[0] * ParticleFilter::refresh_rate;
        theta_y1 = currState.omega[1] * ParticleFilter::refresh_rate;
        theta_z1 = currState.omega[2] * ParticleFilter::refresh_rate;

        theta_y2 = asin(-currState.acceleration[0]/-9.81);
        theta_x2 = atan2(-currState.acceleration[1]/cos(theta_y2),-currState.acceleration[2]/cos(theta_y2));
        theta_z2 = atan2(-currState.magnet[1],currState.magnet[0]);

        for(int i=0; i<3; i++){
            if(i==0){
                theta_xfused = a*theta_x1 + (1-a)*theta_x2;
            }
            if(i==1){
                theta_yfused = a*theta_y1 + (1-a)*theta_y2;
            }

            if(i==2){
                theta_zfused = a*theta_z1 + (1-a)*theta_z2;
            }
        }
        std::cout << "entered initalize particles" << std::endl;
        float currOmegax = 0.0;
        float currOmegay = 0.0;
        float currOmegaz = 0.0;
        float currAlphax = 0.0;
        float currAlphay = 0.0;
        float currAlphaz = 0.0;

        currOmegax = theta_xfused/ParticleFilter::refresh_rate;
        currOmegay = theta_yfused/ParticleFilter::refresh_rate;
        currOmegaz = theta_zfused/ParticleFilter::refresh_rate;

        currAlphax = currOmegax/ParticleFilter::refresh_rate;
        currAlphay = currOmegay/ParticleFilter::refresh_rate;
        currAlphaz = currOmegaz/ParticleFilter::refresh_rate;

        float currVelx =currState.acceleration[0] / ParticleFilter::refresh_rate;
        float currVely = currState.acceleration[1] / ParticleFilter::refresh_rate;
        float currVelz =currState.acceleration[2] / ParticleFilter::refresh_rate;


        //variables in sensor frame, velocity, acceleration, thetas, omegas, alphas
        Eigen::Quaternionf qz(cos(theta_zfused/2),0,0,sin(theta_zfused/2));
        Eigen::Quaternionf qx(cos(theta_xfused/2),sin(theta_xfused/2),0,0);
        Eigen::Quaternionf qy(cos(theta_yfused/2),0,sin(theta_yfused/2),0);
        Eigen::Quaternionf initq;

        initq = qz * qy * qx;

        ParticleFilter::q_normalized = initq.normalized();
        
        // Eigen::Quaternionf velocityGlobal;

        Eigen::Quaternionf velocityGlobal = (q_normalized * Eigen::Quaternionf(0, currVelx,currVely,currVelz) * q_normalized.conjugate());
        Eigen::Quaternionf accelerationsGlobal = (q_normalized * Eigen::Quaternionf(0,currState.acceleration[0],currState.acceleration[1],currState.acceleration[2]) * q_normalized.conjugate());
        Eigen::Quaternionf thetaGlobal = (q_normalized * Eigen::Quaternionf(0, theta_xfused,theta_yfused,theta_zfused) * q_normalized.conjugate());
        Eigen::Quaternionf omegasGlobal = (q_normalized * Eigen::Quaternionf(0, currOmegax,currOmegay,currOmegaz) * q_normalized.conjugate());
        Eigen::Quaternionf alphasGlobal = (q_normalized * Eigen::Quaternionf(0, currAlphax,currAlphay,currAlphaz) * q_normalized.conjugate());

        for(int i=0; i<ParticleFilter::N; i++){
            auto components = randomize();
            Eigen::Vector3f position_vector_Global(currState.position.x(), currState.position.y(), currState.position.z());
            position_vector_Global+= components.rand_position;
            Eigen::Vector3f velocity_vector_Global(velocityGlobal.x(),velocityGlobal.y(),velocityGlobal.z());
            velocity_vector_Global += components.rand_acceleration;
            Eigen::Vector3f accelerations_vector_Global(accelerationsGlobal.x(),accelerationsGlobal.y(),accelerationsGlobal.z());
            accelerations_vector_Global+=components.rand_acceleration;
            Eigen::Vector3f theta_vector_Global(thetaGlobal.x(),thetaGlobal.y(),thetaGlobal.z());
            theta_vector_Global+=components.rand_angular_velocity;
            Eigen::Vector3f omega_vector_Global(omegasGlobal.x(),omegasGlobal.y(),omegasGlobal.z());
            omega_vector_Global+=components.rand_angular_velocity;
            Eigen::Vector3f alpha_vector_Global(alphasGlobal.x(),alphasGlobal.y(),alphasGlobal.z());

            Particle particle(position_vector_Global,velocity_vector_Global,accelerations_vector_Global,
                theta_vector_Global,omega_vector_Global,alpha_vector_Global); 
                  
            ParticleFilter::particles.emplace_back(particle);
            ParticleFilter::particleWeights[i]=1/ParticleFilter::N;
        
        }

        



        //Setting the particles a set of ParticleVectors,, addNoise_and_populate();


        //convert sensor values to world frame


        



        // intialize equally distributed weights
        // q_z = [cos(angle_z/2),0,0,sin(angle_z/2)]
        // q_y = [cos(angle_y/2),0,sin(angle_y/2),0]
        // q_x = [cos(angle_x/2),sin(angle_y/2),0]
        // q^-1 = [q_w,-qx,-qy,-qz]


        // q = [q_w,q_x,q_y,q_z] //p = [q_w,q_x,q_y,q_z]
        // q*p = {[qw*pw-qx*px-qy*py-qz*pz],[qwpx + qxpw + qypz - qzpy],]
        //qp continued [qwpy - qxpz + qypw + qzpx],[qwpz + qxpy - qypx + qzpw]} 
        //then normalize


        //then normalize hoe

        //is this needed?
        //probably..#TODO
        //[-gsin(theta_y), gsin(theta_x)cos(theta_y), -gcos(theta_x)cos(theta_y)]
    }


    void ParticleFilter::ProgressParticles_prelaunch(const StateVector& currState){
        Eigen::Quaternionf qdot;
        Eigen::Quaternionf omega_q;

        float a = 0.4;
        float theta_x1=0.0;
        float theta_x2=0.0;
        float theta_y1=0.0;
        float theta_y2=0.0;
        float theta_z1=0.0;
        float theta_z2=0.0;

        float theta_xfused=0.0;
        float theta_yfused=0.0;
        float theta_zfused=0.0;

        //TODO incorporate divide by 0 error logic by some 1e-6

        theta_x1 = currState.omega[0] * ParticleFilter::refresh_rate;
        theta_y1 = currState.omega[1] * ParticleFilter::refresh_rate;
        theta_z1 = currState.omega[2] * ParticleFilter::refresh_rate;

        theta_y2 = asin(-currState.acceleration[0]/-9.81);
        theta_x2 = atan2(-currState.acceleration[1]/cos(theta_y2),-currState.acceleration[2]/cos(theta_y2));
        theta_z2 = atan2(-currState.magnet[1],currState.magnet[0]);

        for(int i=0; i<3; i++){
            if(i==0){
                theta_xfused = a*theta_x1 + (1-a)*theta_x2;
            }
            if(i==1){
                theta_yfused = a*theta_y1 + (1-a)*theta_y2;
            }

            if(i==2){
                theta_zfused = a*theta_z1 + (1-a)*theta_z2;
            }
        }
        
        float currOmegax = 0.0;
        float currOmegay = 0.0;
        float currOmegaz = 0.0;
        float currAlphax = 0.0;
        float currAlphay = 0.0;
        float currAlphaz = 0.0;

        currOmegax = theta_xfused/ParticleFilter::refresh_rate;
        currOmegay = theta_yfused/ParticleFilter::refresh_rate;
        currOmegaz = theta_zfused/ParticleFilter::refresh_rate;

        currAlphax = currOmegax/ParticleFilter::refresh_rate;
        currAlphay = currOmegay/ParticleFilter::refresh_rate;
        currAlphaz = currOmegaz/ParticleFilter::refresh_rate;
        
        float currVelx = 0.0;
        float currVely = 0.0;
        float currVelz = 0.0;

        currVelx =currState.acceleration[0] / ParticleFilter::refresh_rate;
        currVely =currState.acceleration[1] / ParticleFilter::refresh_rate;
        currVelz =currState.acceleration[2] / ParticleFilter::refresh_rate;

        //ok, here is the methodology we know how to compute our linear dynamics by the accelerometer ez af
        //why did the conventions change? based on the axis evaluated, but theyre just naming conventions. 
        //we just added a new methodology for calculating roll, pitch, and yaw which will come from two sources (integrated gyro(roll(x),pitch(y),yaw(z)), accelerometer: pitch(y), roll(x),yaw(z) magnetometer)
        //we will most likely update the currStateVector's angles by a complementary filter approach of both sensored values, 
        //particles' dynamics will get progressed by vectorized multiplication, helps with compute speed
        
        
        //!!BIG TODO is to find flight approximate variable trusting for magnetometer, accelerometer, gyro

        omega_q = Eigen::Quaternionf(0,currOmegax,currOmegay,currOmegaz);
        qdot = ParticleFilter::q_normalized * omega_q;
        qdot.coeffs() *= 0.5f;
        
        qdot.coeffs() *= ParticleFilter::refresh_rate;
        Eigen::Quaternionf new_q;
        new_q.coeffs() = ParticleFilter::q_normalized.coeffs() + qdot.coeffs();
        ParticleFilter::q_normalized = new_q.normalized();
        
        Eigen::Quaternionf accelerationsGlobal(0,currState.acceleration[0],currState.acceleration[1],currState.acceleration[2]);
        Eigen::Quaternionf velocityGlobal(0, currVelx,currVely,currVelz);
        Eigen::Quaternionf thetaGlobal(0, theta_xfused,theta_yfused,theta_zfused);
        Eigen::Quaternionf omegasGlobal(0, currOmegax,currOmegay,currOmegaz);
        Eigen::Quaternionf alphasGlobal(0, currAlphax,currAlphay,currAlphaz);

        //TODO big question and implication, is this right?
        velocityGlobal = (ParticleFilter::q_normalized * velocityGlobal * q_normalized.conjugate());
        accelerationsGlobal = ParticleFilter::q_normalized * accelerationsGlobal * ParticleFilter::q_normalized.conjugate();
        thetaGlobal = (q_normalized * thetaGlobal * q_normalized.conjugate());
        omegasGlobal = (q_normalized * omegasGlobal * q_normalized.conjugate());
        alphasGlobal = (q_normalized * alphasGlobal * q_normalized.conjugate());

        auto components = randomize();
        Eigen::Vector3f position_vector_Global(currState.position.x(), currState.position.y(), currState.position.z());
        position_vector_Global+= components.rand_position;
        Eigen::Vector3f velocity_vector_Global(velocityGlobal.x(),velocityGlobal.y(),velocityGlobal.z());
        velocity_vector_Global += components.rand_acceleration;
        Eigen::Vector3f accelerations_vector_Global(accelerationsGlobal.x(),accelerationsGlobal.y(),accelerationsGlobal.z());
        accelerations_vector_Global+=components.rand_acceleration;
        Eigen::Vector3f theta_vector_Global(thetaGlobal.x(),thetaGlobal.y(),thetaGlobal.z());
        theta_vector_Global+=components.rand_angular_velocity;
        Eigen::Vector3f omega_vector_Global(omegasGlobal.x(),omegasGlobal.y(),omegasGlobal.z());
        omega_vector_Global+=components.rand_angular_velocity;
        Eigen::Vector3f alpha_vector_Global(alphasGlobal.x(),alphasGlobal.y(),alphasGlobal.z());

        for(Particle& p: ParticleFilter::particles){
            auto components = randomize();
            
        // + .5*(pow((accelerations_vector_Global)*ParticleFilter::refresh_rate,2)?
            
            p.position += velocity_vector_Global *ParticleFilter::refresh_rate;
            p.velocity += accelerations_vector_Global*ParticleFilter::refresh_rate;
            p.accel = accelerations_vector_Global + components.rand_acceleration;
            p.theta+= omega_vector_Global + components.rand_angular_velocity;
            p.omega += (alpha_vector_Global * ParticleFilter::refresh_rate) + components.rand_angular_velocity;
            p.alpha = alpha_vector_Global + components.rand_angular_velocity;
        }
        ParticleFilter::reweightParticles();

    }
        //TODO SEND TO REWEIGHTING!!

        

        //q update is first, []

        
        // q' = .5(q)⊗w_q
        //qt+1 = qt + q'dt
        //[theta = theta + omega(dt) + noise]
        //[omega = complementary omegas from sensor fusion + noise]
        //[alpha = differentiated complementary omegas]
        //[pos = pos + dtvel + 1/2(dt^2)(accel) + noise]
        //[velocities = velocity + accel(dt) + noise]
        //[acceleration = acceleration + weight]

        //roll=x, pitch=y, yaw=z aerospace convention



        //GYROSCOPE
        //omegax
        //omegay
        //omegaz

        //during flight: magnetometer rotation conversion to look up table

        //FROM MAGNETOMETER AND ACCELEROMETER preflight

        //angle_y=arctan_2(a_y,a_z)
        //angle_x = arcsin(a_x/g)
        //angle_z = atan2(-By,Bx)

        //INITIALIZE q based on these integrated angle sensor values 
        //q_w=cos(θx/2)cos(θy/2)cos(θz/2) + sin(θx/2)sin(θy/2)sin(θz/2)
        //q_x = cos(θx/2)cos(θy/2)cos(θz/2) - sin(θx/2)sin(θy/2)sin(θz/2)
        //q_y = cos(θx/2)sin(θy/2)cos(θz/2) + sin(θx/2)cos(θy/2)sin(θz/2)
        //q_z = cos(θx/2)cos(θy/2)sin(θz/2) + sin(θx/2)cos(θy/2)sin(θz/2)

        //2. convert accels, thetas to 

    void ParticleFilter::ProgressParticles_postlaunch(const StateVector& currState){
        Eigen::Quaternionf omega_q;
        Eigen::Quaternionf qdot;
        float a = 0.4;
        float theta_x1=0.0;
        float theta_x2=0.0;
        float theta_y1=0.0;
        float theta_y2=0.0;
        float theta_z1=0.0;

        float theta_xfused=0.0;
        float theta_yfused=0.0;
        float theta_zfused=0.0;
        float theta_z2 = atan2(-currState.magnet[1],currState.magnet[0]);
        //edit sensor fusion of angles to be magnetometer!

        theta_x1 = currState.omega[0] * ParticleFilter::refresh_rate;
        theta_y1 = currState.omega[1] * ParticleFilter::refresh_rate;
        theta_z1 = currState.omega[2] * ParticleFilter::refresh_rate;

        //TODO: theta 2 representation as magnetometer, reference operations,
            //Options, add baked in noise from preFlight variance
            //Options, compute by rotational matrix the difference between the rotational difference of 
            //the magnetic vector and the field vector, normalize them prior, pre initialize to prev refresh rate's 
            //euler angles to get a better time complexity for reaching the desired frame
            //

        //in z, y, x, done by rotational frames, but what about from quaternion?

        for(int i=0; i<3; i++){
            if(i==0){
                theta_xfused = theta_x1; /* + (1-a)*theta_x2; */
            }
            if(i==1){
                theta_yfused = theta_y1;/* + (1-a)*theta_y2; */
            }

            if(i==2){
                theta_zfused = a* theta_z1  + (1-a)*theta_z2; 
            
        }
        
        float currOmegax = 0.0;
        float currOmegay = 0.0;
        float currOmegaz = 0.0;
        float currAlphax = 0.0;
        float currAlphay = 0.0;
        float currAlphaz = 0.0;

        currOmegax = theta_xfused/ParticleFilter::refresh_rate;
        currOmegay = theta_yfused/ParticleFilter::refresh_rate;
        currOmegaz = theta_zfused/ParticleFilter::refresh_rate;

        currAlphax = currOmegax/ParticleFilter::refresh_rate;
        currAlphay = currOmegay/ParticleFilter::refresh_rate;
        currAlphaz = currOmegaz/ParticleFilter::refresh_rate;
        
        float currVelx = 0.0;
        float currVely = 0.0;
        float currVelz = 0.0;

        currVelx =currState.acceleration[0] / ParticleFilter::refresh_rate;
        currVely =currState.acceleration[1] / ParticleFilter::refresh_rate;
        currVelz =currState.acceleration[2] / ParticleFilter::refresh_rate;

        omega_q = Eigen::Quaternionf(0,currOmegax,currOmegay,currOmegaz);
        qdot = ParticleFilter::q_normalized * omega_q;
        qdot.coeffs() *= 0.5f;
        
        qdot.coeffs() *= ParticleFilter::refresh_rate;
        Eigen::Quaternionf new_q;
        new_q.coeffs() = ParticleFilter::q_normalized.coeffs() + qdot.coeffs();
        ParticleFilter::q_normalized = new_q.normalized();
        
        Eigen::Quaternionf accelerationsGlobal(0,currState.acceleration[0],currState.acceleration[1],currState.acceleration[2]);
        Eigen::Quaternionf velocityGlobal(0, currVelx,currVely,currVelz);
        Eigen::Quaternionf thetaGlobal(0, theta_xfused,theta_yfused,theta_zfused);
        Eigen::Quaternionf omegasGlobal(0, currOmegax,currOmegay,currOmegaz);
        Eigen::Quaternionf alphasGlobal(0, currAlphax,currAlphay,currAlphaz);

        //TODO big question and implication, is this right?
        velocityGlobal = (ParticleFilter::q_normalized * velocityGlobal * q_normalized.conjugate());
        accelerationsGlobal = ParticleFilter::q_normalized * accelerationsGlobal * ParticleFilter::q_normalized.conjugate();
        thetaGlobal = (q_normalized * thetaGlobal * q_normalized.conjugate());
        omegasGlobal = (q_normalized * omegasGlobal * q_normalized.conjugate());
        alphasGlobal = (q_normalized * alphasGlobal * q_normalized.conjugate());

        auto components = randomize();
        Eigen::Vector3f position_vector_Global(currState.position.x(), currState.position.y(), currState.position.z());
        position_vector_Global+= components.rand_position;
        Eigen::Vector3f velocity_vector_Global(velocityGlobal.x(),velocityGlobal.y(),velocityGlobal.z());
        velocity_vector_Global += components.rand_acceleration;
        Eigen::Vector3f accelerations_vector_Global(accelerationsGlobal.x(),accelerationsGlobal.y(),accelerationsGlobal.z());
        accelerations_vector_Global+=components.rand_acceleration;
        Eigen::Vector3f theta_vector_Global(thetaGlobal.x(),thetaGlobal.y(),thetaGlobal.z());
        theta_vector_Global+=components.rand_angular_velocity;
        Eigen::Vector3f omega_vector_Global(omegasGlobal.x(),omegasGlobal.y(),omegasGlobal.z());
        omega_vector_Global+=components.rand_angular_velocity;
        Eigen::Vector3f alpha_vector_Global(alphasGlobal.x(),alphasGlobal.y(),alphasGlobal.z());

        for(Particle& p: ParticleFilter::particles){
            auto components = randomize();
        // + .5*(pow((accelerations_vector_Global)*ParticleFilter::refresh_rate,2)?
            
            p.position += velocity_vector_Global *ParticleFilter::refresh_rate;
            p.velocity += accelerations_vector_Global*ParticleFilter::refresh_rate;
            p.accel = accelerations_vector_Global + components.rand_acceleration;
            p.theta+= omega_vector_Global + components.rand_angular_velocity;
            p.omega += (alpha_vector_Global * ParticleFilter::refresh_rate) + components.rand_angular_velocity;
            p.alpha = alpha_vector_Global + components.rand_angular_velocity;
        }
        ParticleFilter::reweightParticles();


                
        //!!BIG TODO is to find flight approximate variable trusting for magnetometer, accelerometer, gyro


        //TODO sensor fusion's angles based on magnetometer calculations!

        }
    }

    void ParticleFilter::resampleParticles(){
        //resample logic should be if 2N/3 > sum of squared weights,
        //n particles should be repopulated based on cdf array but this will result in a very poor time complexity 
        //stratified sample of N buckets from 0-1 where each bucket length is 1/N
            //ordered based value so, go until ith sample has been found
            //cdf should still be formed by the range of the weights 
        float CumSum = 0.0;
        for(int i=0; i<ParticleFilter::N;i++){
            CumSum+=pow(ParticleFilter::particleWeights[i],2);
        }
        
        if (2*ParticleFilter::N/3 > CumSum){
            std::vector<Particle> newParticles;
            std::random_device rd;  // Seed for the random number engine
            std::mt19937 gen(rd());
            float samples[ParticleFilter::N];
            float cdfArray[ParticleFilter::N];
            float cumCounter=0.0;
            float cumTill=0.0;
            float random_number = 0.0;

            //stratified sampler in order!
            for(int i=0; i<ParticleFilter::N;i++){
                
                if (i<ParticleFilter::N-1){
                    cumCounter = i/ParticleFilter::N;
                    cumTill = (i+1)/ParticleFilter::N;
                    std::uniform_real_distribution<> dis(cumCounter, cumTill);
                    random_number = dis(gen);
                    }
                else{
                    std::uniform_real_distribution<> dis(cumTill, 1);
                    random_number = dis(gen);    
                }

                samples[i]=random_number;
            }

            for(int i=0; i<ParticleFilter::N;i++){
                if (i==0){
                    cdfArray[i]=ParticleFilter::particleWeights[i];
                    }
                else{
                    cdfArray[i]+=cdfArray[i-1]+ParticleFilter::particleWeights[i];
                }
            }

            // Normalize CDF to end at 1.0
            float cdfMax = cdfArray[(ParticleFilter::N)-1];
            
            for(int i=0; i<ParticleFilter::N; i++) {
                cdfArray[i] /= cdfMax;
            }

            for(int i=0; i<ParticleFilter::N; i++) {
                // Binary search to find the right particle
                int index = 0;
                while(index < ParticleFilter::N-1 && cdfArray[index] < samples[i]) {
                    index++;
                }
                
                // Add the selected particle to new set
                newParticles.push_back(ParticleFilter::particles[index]);
                particles=newParticles;
            }
        reweightParticles();
        }    
    }

    void ParticleFilter::reweightParticles(){
        float currLossArr[ParticleFilter::N];
        //weight normalization
        float currLoss = 0.0;
        float weightSum = 0.0;
        float currWeightCalc = 0.0;
        float currWeightNum = 0.0;
        int index = 0;

        // for(Particle &p : ParticleFilter::particles){
        //     Particle differenceVector=p -ParticleFilter::StateParticle;
        //     currLoss=differenceVector.T * differenceVector;
        //     weightSum+=std::exp(-.5*currLoss);
        // }

        int i=0; 

        for(Particle& p: ParticleFilter::particles){
            Eigen::Vector3f position_diff = p.position - StateParticle.position;
            Eigen::Vector3f velocity_diff = p.velocity - StateParticle.velocity;
            Eigen::Vector3f acceleration_diff = p.accel - StateParticle.accel;
            Eigen::Vector3f orientation_diff = p.theta - StateParticle.theta;
            Eigen::Vector3f omega_diff = p.omega - StateParticle.omega;
            Eigen::Vector3f alpha_diff = p.alpha - StateParticle.alpha;

            float currLoss = position_diff.squaredNorm() + velocity_diff.squaredNorm() + 
                 acceleration_diff.squaredNorm() + orientation_diff.squaredNorm() + 
                 omega_diff.squaredNorm() + alpha_diff.squaredNorm();
                 currLossArr[i]=std::exp(-.5*currLoss);
                 weightSum+=currLossArr[i];
            i+=1;
        }

        for(int j=0; j<ParticleFilter::N; j++){
            ParticleFilter::particleWeights[j]=currLossArr[j]/weightSum;
        }

// Compute loss (squared Euclidean distance for Vector3f, and angle difference for Quaternionf)

            // Particle differenceVector=p-ParticleFilter::StateParticle;
            // currLoss=differenceVector.T * differenceVector;
        
        //Make matching checkerVector by currState Vector's values and particle's shape.

        //Precompute weightSum =Σ e^(-.5(||currData - currParticle||^2))

        //compute each particles' weight by (e^(-.5||output-currParticle||^2))/weightSum
    }

    void ParticleFilter::filterParticles(){
        std::vector<size_t> indices(ParticleFilter::N);
        std::iota(indices.begin(), indices.end(), 0); 

        std::sort(indices.begin(), indices.end(), [this](size_t i, size_t j) {
            return ParticleFilter::particleWeights[i] < ParticleFilter::particleWeights[j]; // Sort descending by weight
        });

            //TODO figure out proper reweighting technique
            int cumSum = 0;
            int threshold = .1;
            int index = 0;
            float cdfArray[ParticleFilter::N];
        
            for(int i=0; i<N; i++){        
                if( cumSum >= threshold){
                    break;
                }
        
                cumSum+=ParticleFilter::particleWeights[indices[i]];
                index+=1;
            }

            std::vector<float> sampleArray(index);
        
            // :index will be resampled upon uniform distribution of cdfArray
            for(int i=0; i<N; i++){
                if(i==0){
                    cdfArray[i]=ParticleFilter::particleWeights[i];
                }

                else{
                    cdfArray[i]+=(ParticleFilter::particleWeights[i] + cdfArray[i-1]);
                }
            }
        
            for(int i=0; i<index; i++){
                double sample = (double)rand() / RAND_MAX;
                sampleArray[i]=sample;
            }
        
            for(int i=0; i<index; i++){

                for(int j=0; j<N; j++){
                    double sample = (double)rand() / RAND_MAX;
                    if(cdfArray[j]>=sampleArray[i]){
                        
                        //
                        ParticleFilter::particles[indices[i]]=ParticleFilter::particles[indices[j]];
                        ParticleFilter::particleWeights[indices[i]]=ParticleFilter::particleWeights[indices[j]];
                        break;
                    }
                }
            }

            ParticleFilter::reweightParticles();
    }

    Particle ParticleFilter::returnParticle(){
        Particle tempStorer;

        for(Particle& p : ParticleFilter::particles){
            for(int i =0; i<19;i++){
                tempStorer.position+=(p.position)/ParticleFilter::N;
                tempStorer.velocity+=(p.velocity)/ParticleFilter::N;
                tempStorer.accel+=(p.accel)/ParticleFilter::N;
                tempStorer.theta+=(p.theta)/ParticleFilter::N;
                tempStorer.omega+=(p.omega)/ParticleFilter::N;
                tempStorer.alpha+=(p.alpha)/ParticleFilter::N;

            }
        }
        return tempStorer;

        }


int main() {
    ParticleFilter pf;
    std::cout << "Hello worl!" << std::endl;
    StateVector test_vector{
        test_vector.position = Eigen::Vector3f(1.0f, 2.0f, 3.0f),       // x, y, z in meters
        test_vector.acceleration = Eigen::Vector3f(0.5f, -0.2f, 9.81f),  // m/s² (including gravity)
        test_vector.omega = Eigen::Vector3f(0.1f, -0.05f, 0.2f),         // Angular velocity (rad/s)
        test_vector.magnet = Eigen::Vector3f(0.3f, 0.5f, -0.4f)          // Magnetic field (Gauss)
    };
    pf.InitializeParticles(test_vector);
    pf.reweightParticles();
    pf.ProgressParticles_prelaunch(test_vector);
    pf.ProgressParticles_postlaunch(test_vector);
    pf.resampleParticles();
    Particle p=pf.returnParticle();
    try{
        //tester
        std::cout <<"test created successfully" <<std::endl;
    }
    catch (const std::exception& e) {  // Catch standard exceptions
        std::cerr << "Error in constructing Particle Filter's state vector Object: " << e.what() << std::endl;
    } 
    std::cout << "Hello world!" << std::endl;
    return 0;
}
        



//ASSUMED STATE VECTOR REPRESENTATION

