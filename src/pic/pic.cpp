#include "pic.h"

#include "wrapper.h"

#include "fields.h"
#include "sim_params.h"
#include "timer.h"

#include <plog/Log.h>

namespace ntt {

  template <Dimension D>
  void PIC<D>::Run() {
    // register the content of em fields
    Simulation<D, PICEngine>::Initialize();
    Simulation<D, PICEngine>::Verify();
    {
      auto  params = *(this->params());
      auto& mblock = this->meshblock;
      auto  timax  = (unsigned long)(params.totalRuntime() / mblock.timestep());

      ResetSimulation();
      Simulation<D, PICEngine>::PrintDetails();
      InitialStep();
      for (unsigned long ti { 0 }; ti < timax; ++ti) {
        PLOGD << "t = " << this->m_time;
        PLOGD << "ti = " << this->m_tstep;
        StepForward();
      }
      WaitAndSynchronize();
    }
    Simulation<D, PICEngine>::Finalize();
  }

  template <Dimension D>
  void PIC<D>::InitialStep() {
    auto& mblock         = this->meshblock;
    mblock.em_content[0] = Content::ex1_cntrv;
    mblock.em_content[1] = Content::ex2_cntrv;
    mblock.em_content[2] = Content::ex3_cntrv;
    mblock.em_content[3] = Content::bx1_cntrv;
    mblock.em_content[4] = Content::bx2_cntrv;
    mblock.em_content[5] = Content::bx3_cntrv;
  }

  template <Dimension D>
  void PIC<D>::Benchmark() {
    Faraday();
    Ampere();
    Faraday();
  }

  template <Dimension D>
  void PIC<D>::StepForward() {
    auto                       params = *(this->params());
    auto&                      mblock = this->meshblock;
    auto&                      wrtr   = this->writer;
    auto&                      pgen   = this->problem_generator;

    timer::Timers              timers({ "FieldSolver",
                                        "FieldBoundaries",
                                        "CurrentDeposit",
                                        "ParticlePusher",
                                        "ParticleBoundaries",
                                        "UserSpecific",
                                        "Output" });
    static std::vector<double> dead_fractions = {};

    if (params.fieldsolverEnabled()) {
      timers.start("FieldSolver");
      Faraday();
      timers.stop("FieldSolver");

      timers.start("FieldBoundaries");
      FieldsExchange();
      FieldsBoundaryConditions();
      timers.stop("FieldBoundaries");
    }

    {
      timers.start("ParticlePusher");
      ParticlesPush();
      timers.stop("ParticlePusher");

      timers.start("UserSpecific");
      // !TODO: this needs to move (or become optional)
      this->ComputeDensity(0);
      pgen.UserDriveParticles(this->m_time, params, mblock);
      timers.stop("UserSpecific");

      timers.start("ParticleBoundaries");
      ParticlesBoundaryConditions();
      timers.stop("ParticleBoundaries");

      if (params.depositEnabled()) {
        timers.start("CurrentDeposit");
        ResetCurrents();
        CurrentsDeposit();

        timers.start("FieldBoundaries");
        CurrentsSynchronize();
        CurrentsExchange();
        CurrentsBoundaryConditions();
        timers.stop("FieldBoundaries");

        CurrentsFilter();
        timers.stop("CurrentDeposit");
      }

      timers.start("ParticleBoundaries");
      ParticlesExchange();
      if ((params.shuffleInterval() > 0) && (this->m_tstep % params.shuffleInterval() == 0)) {
        dead_fractions = mblock.RemoveDeadParticles(params.maxDeadFraction());
      }
      timers.stop("ParticleBoundaries");
    }

    if (params.fieldsolverEnabled()) {
      timers.start("FieldSolver");
      Faraday();
      timers.stop("FieldSolver");

      timers.start("FieldBoundaries");
      FieldsExchange();
      FieldsBoundaryConditions();
      timers.stop("FieldBoundaries");

      timers.start("FieldSolver");
      Ampere();
      timers.stop("FieldSolver");

      if (params.depositEnabled()) {
        timers.start("CurrentDeposit");
        AmpereCurrents();
        timers.stop("CurrentDeposit");
      }

      timers.start("FieldBoundaries");
      FieldsExchange();
      FieldsBoundaryConditions();
      timers.stop("FieldBoundaries");
    }

    timers.start("Output");
    if ((params.outputFormat() != "disabled")
        && (this->m_tstep % params.outputInterval() == 0)) {
      WaitAndSynchronize();
      ComputeDensity();
      this->SynchronizeHostDevice();
      InterpolateAndConvertFieldsToHat();
      wrtr.WriteFields(params, mblock, this->m_time, this->m_tstep);
    }
    timers.stop("Output");

    timers.printAll("time = " + std::to_string(this->m_time)
                    + " : timestep = " + std::to_string(this->m_tstep));
    this->PrintDiagnostics(std::cout, dead_fractions);

    this->m_time += mblock.timestep();
    this->m_tstep++;
  }

  template <Dimension D>
  void PIC<D>::StepBackward() {}
}    // namespace ntt

template class ntt::PIC<ntt::Dim1>;
template class ntt::PIC<ntt::Dim2>;
template class ntt::PIC<ntt::Dim3>;
