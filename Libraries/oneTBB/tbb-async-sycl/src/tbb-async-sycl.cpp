//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <cmath>  //for std::ceil
#include <array>
#include <atomic>
#include <iostream>
#include <thread>
#include <numeric>

#include <CL/sycl.hpp>

#include <tbb/blocked_range.h>
#include <tbb/flow_graph.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

struct done_tag{};

const float ratio = 0.5;  // CPU to GPU offload ratio
const float alpha = 0.5;  // coeff for triad calculation

const size_t array_size = 16;

template<class T>
void PrintArr(const char* text, const T& array) {
  std::cout << text;
  for (const auto& s : array) std::cout << s << ' ';
  std::cout << "\n";
}

int main() {

  sycl::queue q(sycl::default_selector{}, dpc_common::exception_handler);

  // USM allocator for data of type int in shared memory
  typedef sycl::usm_allocator<float, sycl::usm::alloc::shared> vec_alloc;
  // Create allocator for device associated with q
  vec_alloc myAlloc(q);
  // Create std vectors with the allocator
  std::vector<float, vec_alloc >
  a_array(array_size, myAlloc),
  b_array(array_size, myAlloc),
  c_array(array_size, myAlloc);

  // init input arrays
  std::iota(a_array.begin(), a_array.end(), 0.0f);
  std::iota(b_array.begin(), b_array.end(), 0.0f);

  int nth = 4; // number of threads

  auto mp = tbb::global_control::max_allowed_parallelism;

  tbb::global_control gc(mp, nth /* + 1*/);  // No need for additional (sleeping) thread!
  tbb::flow::graph g;

  // Input node:
  tbb::flow::input_node<float> in_node{g,
                                        [&](tbb::flow_control& fc) -> float {
                                          static bool has_run = false;
                                          if (has_run) fc.stop();
                                          has_run = true;
                                          return ratio;
                                        }
                                      };

  // CPU node
  tbb::flow::function_node<float, done_tag> cpu_node{
      g, tbb::flow::unlimited, [&](float offload_ratio) {
        size_t i_start = static_cast<size_t>(std::ceil(array_size * offload_ratio));
        size_t i_end = static_cast<size_t>(array_size);
        std::cout << "start index for CPU = " << i_start
                  << "; end index for CPU = " << i_end << "\n";

        tbb::parallel_for(tbb::blocked_range<size_t>{i_start, i_end},
                          [&](const tbb::blocked_range<size_t>& r) {
                            for (size_t i = r.begin(); i < r.end(); ++i)
                              c_array[i] = a_array[i] + alpha * b_array[i];
                          });
        return done_tag{};
      }};

  using async_node_type = tbb::flow::async_node<float, done_tag>;
  using gateway_type = async_node_type::gateway_type;

  // async node -- GPU
  async_node_type a_node{
      g, tbb::flow::unlimited,
      [&](const float& offload_ratio, gateway_type& gateway) {
        gateway.reserve_wait();
        //async_act.submit(offload_ratio, gateway);
        size_t array_size_sycl = std::ceil(array_size * offload_ratio);
        std::cout << "start index for GPU = 0; end index for GPU = "
                  << array_size_sycl << "\n";
        const float coeff = alpha;  // coeff is a local varaible

        // Get pointer to vector data for access in kernel
        auto A = a_array.data();
        auto B = b_array.data();
        auto C = c_array.data();

        auto event = q.submit([&](sycl::handler& h) {
            h.parallel_for( sycl::range<1>{array_size_sycl}, [=](sycl::id<1> index) {
              C[index] = A[index] + B[index] * coeff;
            });  // end of the kernel -- parallel for
          });
          // enqueue host task that notifies flow graph that device is done! 
          q.submit([&](sycl::handler& h) {
            h.depends_on(event);
            h.host_task([&](){
              gateway.try_put(done_tag{});
              gateway.release_wait();
            });
          });
        }
      }};

  // join node
  using join_t =
      tbb::flow::join_node<std::tuple<done_tag, done_tag>, tbb::flow::queueing>;
  join_t node_join{g};

  // out node
  tbb::flow::function_node<join_t::output_type> out_node{
      g, tbb::flow::unlimited, [&](const join_t::output_type&) {
        // Serial execution
        std::array<float, array_size> c_gold;
        for (size_t i = 0; i < array_size; ++i)
          c_gold[i] = a_array[i] + alpha * b_array[i];

        // Compare golden triad with heterogeneous triad
        if (!std::equal(std::begin(c_array), std::end(c_array),
                        std::begin(c_gold)))
          std::cout << "Heterogenous triad error.\n";
        else
          std::cout << "Heterogenous triad correct.\n";

        PrintArr("c_array: ", c_array);
        PrintArr("c_gold : ", c_gold);
      }};  // end of out node

  // construct graph
  tbb::flow::make_edge(in_node, a_node);
  tbb::flow::make_edge(in_node, cpu_node);
  tbb::flow::make_edge(a_node, tbb::flow::input_port<0>(node_join));
  tbb::flow::make_edge(cpu_node, tbb::flow::input_port<1>(node_join));
  tbb::flow::make_edge(node_join, out_node);

  in_node.activate();
  g.wait_for_all();

  return 0;
}
