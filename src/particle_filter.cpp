/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

inline double sqr(double x) {
  return x * x;
}

inline double sqr_dist(double x1, double y1, double x2, double y2) {
  return sqr(x2 - x1) + sqr(y2 - y1);
}

inline double multivariate_gaussian(double x, double y, double ux, double uy, double sx, double sy) {
  return exp(-(sqr(x - ux) / sqr(sx) + sqr(y - uy) / sqr(sy))/2) / (2 * M_PI * sx * sy);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 100;
  particles.reserve(num_particles);
  weights.reserve(num_particles);

  default_random_engine generator;
  normal_distribution<double> xdist(x, std[0]);
  normal_distribution<double> ydist(y, std[1]);
  normal_distribution<double> thetadist(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = 0;
    particle.x = xdist(generator);
    particle.y = ydist(generator);
    particle.theta = thetadist(generator);
    particle.weight = 0.0;
    particles.push_back(particle);
    weights.push_back(0.0);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine generator;

  for (int i = 0; i < num_particles; i++) {

    double thetaf, xf, yf;
    if (abs(yaw_rate) > 0.0001) {
      thetaf = particles[i].theta + yaw_rate * delta_t;
      xf = particles[i].x + velocity * (sin(thetaf) - sin(particles[i].theta)) / yaw_rate;
      yf = particles[i].y + velocity * (cos(particles[i].theta) - cos(thetaf)) / yaw_rate;
    }
    else {
      thetaf = particles[i].theta;
      xf = particles[i].x + velocity * delta_t * cos(thetaf);
      yf = particles[i].y + velocity * delta_t * sin(thetaf);
    }

    normal_distribution<double> xdist(xf, std_pos[0]);
    normal_distribution<double> ydist(yf, std_pos[1]);
    normal_distribution<double> thetadist(thetaf, std_pos[2]);

    particles[i].x = xdist(generator);
    particles[i].y = ydist(generator);
    particles[i].theta = thetadist(generator);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  for (int i = 0; i < observations.size(); i++) {
    double min_sqr_dist = numeric_limits<double>::max();
    for (int j = 0; j < predicted.size(); j++) {
      double sd = sqr_dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (sd < min_sqr_dist) {
        observations[i].id = j;
        min_sqr_dist = sd;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {

  for (int i = 0; i < num_particles; i++) {

    // 1. Transform observations to real world coordinates
    double xp = particles[i].x;
    double yp = particles[i].y;
    double thetap = particles[i].theta;

    std::vector<LandmarkObs> obs_world = observations;
    for (int j = 0; j < observations.size(); j++) {
      double xo = observations[j].x;
      double yo = observations[j].y;

      obs_world[j].x = xo * cos(thetap) - yo * sin(thetap) + xp;
      obs_world[j].y = xo * sin(thetap) + yo * cos(thetap) + yp;
    }

    // 2. Find landmarks in range
    std::vector<LandmarkObs> predicted;
    predicted.reserve(20);

    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      double xm = (double)map_landmarks.landmark_list[j].x_f;
      double ym = (double)map_landmarks.landmark_list[j].y_f;

      if (sqr_dist(xm, ym, xp, yp) <= sqr(sensor_range)) {
        LandmarkObs obs;
        obs.x = xm;
        obs.y = ym;
        obs.id = map_landmarks.landmark_list[j].id_i;
        predicted.push_back(obs);
      }
    }

    if (predicted.size() != 0) {
      // 3. Associate landmarks with observations using nearest-neighbors
      dataAssociation(predicted, obs_world);

      // 4. Calculate final weight using Multivariate-Gaussian probability
      weights[i] = 1;
      for (int j = 0; j < obs_world.size(); j++) {
        double xo = obs_world[j].x;
        double yo = obs_world[j].y;
        double xr = predicted[obs_world[j].id].x;
        double yr = predicted[obs_world[j].id].y;

        weights[i] *= multivariate_gaussian(xo, yo, xr, yr, std_landmark[0], std_landmark[1]);
      }

      particles[i].weight = weights[i];
    }
  }
}

void ParticleFilter::resample() {
  default_random_engine generator;
  discrete_distribution<int> ddist(weights.begin(), weights.end());

  vector<Particle> particles_copy = particles;
  for (int i = 0; i < num_particles; i++) {
    particles[i] = particles_copy[ddist(generator)];
  }
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
