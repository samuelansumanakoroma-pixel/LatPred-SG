NS-3 simulation script:
```cpp
/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include <fstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("SmartGridSim");

// Custom tag to carry priority (0 = lowest, 7 = highest)
class PriorityTag : public Tag
{
public:
  static TypeId GetTypeId(void);
  virtual TypeId GetInstanceTypeId(void) const;
  virtual uint32_t GetSerializedSize(void) const;
  virtual void Serialize(Buffer::Iterator i) const;
  virtual void Deserialize(Buffer::Iterator i);
  virtual void Print(std::ostream &os) const;

  void SetPriority(uint8_t p) { m_priority = p; }
  uint8_t GetPriority(void) const { return m_priority; }
private:
  uint8_t m_priority = 0;
};

NS_OBJECT_ENSURE_REGISTERED(PriorityTag);

TypeId PriorityTag::GetTypeId(void)
{
  static TypeId tid = TypeId("ns3::PriorityTag")
    .SetParent<Tag>()
    .SetGroupName("Applications")
    .AddConstructor<PriorityTag>();
  return tid;
}
TypeId PriorityTag::GetInstanceTypeId(void) const { return GetTypeId(); }
uint32_t PriorityTag::GetSerializedSize(void) const { return 1; }
void PriorityTag::Serialize(Buffer::Iterator i) const { i.WriteU8(m_priority); }
void PriorityTag::Deserialize(Buffer::Iterator i) { m_priority = i.ReadU8(); }
void PriorityTag::Print(std::ostream &os) const { os << "Priority=" << (uint32_t)m_priority; }

int main(int argc, char *argv[])
{
  // Simulation parameters
  uint32_t nMeters = 50;
  uint32_t nGateways = 5;
  uint32_t nSubstations = 2;
  double simTime = 60.0;
  double faultStart = 20.0;
  double faultDuration = 5.0;
  int backgroundLoad = 0;  // 0, 20, 50 (percentage)

  // Link parameters
  std::string hanLinkRate = "10Mbps";
  std::string hanLinkDelay = "2ms";
  std::string nanLinkRate = "100Mbps";
  std::string nanLinkDelay = "5ms";
  std::string wanLinkRate = "100Mbps";
  std::string wanLinkDelay = "10ms";

  CommandLine cmd;
  cmd.AddValue("nMeters", "Number of smart meters", nMeters);
  cmd.AddValue("nGateways", "Number of aggregation gateways", nGateways);
  cmd.AddValue("nSubstations", "Number of substations", nSubstations);
  cmd.AddValue("simTime", "Simulation time (s)", simTime);
  cmd.AddValue("faultStart", "Fault burst start time (s)", faultStart);
  cmd.AddValue("faultDuration", "Fault burst duration (s)", faultDuration);
  cmd.AddValue("backgroundLoad", "Background load percentage (0,20,50)", backgroundLoad);
  cmd.Parse(argc, argv);

  // Enable priority queue disc
  Config::SetDefault("ns3::PfifoFastQueueDisc::MaxSize", QueueSizeValue(QueueSize("1000p")));

  // Create nodes
  NodeContainer meters;
  meters.Create(nMeters);
  NodeContainer gateways;
  gateways.Create(nGateways);
  NodeContainer substations;
  substations.Create(nSubstations);
  NodeContainer controlCenter;
  controlCenter.Create(1);

  // --- Connect meters to gateways ---
  PointToPointHelper p2pHan;
  p2pHan.SetDeviceAttribute("DataRate", StringValue(hanLinkRate));
  p2pHan.SetChannelAttribute("Delay", StringValue(hanLinkDelay));

  uint32_t metersPerGateway = nMeters / nGateways;
  Ipv4InterfaceContainer meterIfs[nGateways];
  Ipv4InterfaceContainer gatewayIfs[nGateways];

  Ipv4AddressHelper ip;
  ip.SetBase("10.1.0.0", "255.255.0.0");

  for (uint32_t gw = 0; gw < nGateways; gw++)
  {
    for (uint32_t m = 0; m < metersPerGateway; m++)
    {
      uint32_t meterIndex = gw * metersPerGateway + m;
      NodeContainer pair(meters.Get(meterIndex), gateways.Get(gw));
      NetDeviceContainer nd = p2pHan.Install(pair);
      ip.NewNetwork();
      Ipv4InterfaceContainer ifs = ip.Assign(nd);
      // Store for future reference if needed
    }
  }

  // --- Connect gateways to substations ---
  PointToPointHelper p2pNan;
  p2pNan.SetDeviceAttribute("DataRate", StringValue(nanLinkRate));
  p2pNan.SetChannelAttribute("Delay", StringValue(nanLinkDelay));

  for (uint32_t gw = 0; gw < nGateways; gw++)
  {
    for (uint32_t sub = 0; sub < nSubstations; sub++)
    {
      NodeContainer pair(gateways.Get(gw), substations.Get(sub));
      NetDeviceContainer nd = p2pNan.Install(pair);
      ip.NewNetwork();
      ip.Assign(nd);
    }
  }

  // --- Connect substations to control center ---
  PointToPointHelper p2pWan;
  p2pWan.SetDeviceAttribute("DataRate", StringValue(wanLinkRate));
  p2pWan.SetChannelAttribute("Delay", StringValue(wanLinkDelay));

  Ipv4InterfaceContainer subCcIfs[nSubstations];
  for (uint32_t sub = 0; sub < nSubstations; sub++)
  {
    NodeContainer pair(substations.Get(sub), controlCenter.Get(0));
    NetDeviceContainer nd = p2pWan.Install(pair);
    ip.NewNetwork();
    subCcIfs[sub] = ip.Assign(nd);
  }

  // --- Install Internet stack ---
  InternetStackHelper stack;
  stack.Install(meters);
  stack.Install(gateways);
  stack.Install(substations);
  stack.Install(controlCenter);

  // --- Global routing ---
  Ipv4GlobalRoutingHelper::PopulateRoutingTables();

  // Get control center IP (first address on its first interface)
  Ipv4Address controlCenterIp = controlCenter.Get(0)->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();

  // --- 1. Periodic monitoring traffic (low priority) ---
  uint16_t monitoringPort = 10000;
  for (uint32_t m = 0; m < nMeters; m++)
  {
    OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(controlCenterIp, monitoringPort));
    onoff.SetAttribute("DataRate", StringValue("100kbps"));
    onoff.SetAttribute("PacketSize", UintegerValue(256));
    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=0.1]"));
    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.9]"));
    ApplicationContainer app = onoff.Install(meters.Get(m));
    app.Start(Seconds(0.0));
    app.Stop(Seconds(simTime));
    // Tag with priority 0
    Ptr<Application> appPtr = app.Get(0);
    appPtr->TraceConnectWithoutContext("Tx", MakeCallback([=](Ptr<const Packet> p) {
        auto packet = const_cast<Packet*>(p);
        PriorityTag tag;
        tag.SetPriority(0);
        packet->AddPacketTag(tag);
      }));
  }

  // --- 2. Fault burst traffic (high priority) ---
  uint16_t faultPort = 20000;
  for (uint32_t m = 0; m < nMeters; m++)
  {
    OnOffHelper fault("ns3::UdpSocketFactory", InetSocketAddress(controlCenterIp, faultPort));
    fault.SetAttribute("DataRate", StringValue("5Mbps"));
    fault.SetAttribute("PacketSize", UintegerValue(512));
    fault.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=0.5]"));
    fault.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.5]"));
    ApplicationContainer faultApp = fault.Install(meters.Get(m));
    faultApp.Start(Seconds(faultStart));
    faultApp.Stop(Seconds(faultStart + faultDuration));
    // Tag with priority 7
    Ptr<Application> faultPtr = faultApp.Get(0);
    faultPtr->TraceConnectWithoutContext("Tx", MakeCallback([=](Ptr<const Packet> p) {
        auto packet = const_cast<Packet*>(p);
        PriorityTag tag;
        tag.SetPriority(7);
        packet->AddPacketTag(tag);
      }));
  }

  // --- 3. Background cross-traffic (low priority) ---
  if (backgroundLoad > 0)
  {
    double bgRate = (double)backgroundLoad / 100.0 * 100.0; // Mbps
    std::string bgRateStr = std::to_string(bgRate) + "Mbps";
    uint16_t bgPort = 30000;
    Ipv4Address sub1Ip = substations.Get(0)->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
    Ipv4Address sub2Ip = substations.Get(1)->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();

    OnOffHelper bg("ns3::UdpSocketFactory", InetSocketAddress(sub2Ip, bgPort));
    bg.SetAttribute("DataRate", StringValue(bgRateStr));
    bg.SetAttribute("PacketSize", UintegerValue(512));
    bg.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=0.5]"));
    bg.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.5]"));
    ApplicationContainer bgApp = bg.Install(substations.Get(0));
    bgApp.Start(Seconds(0.0));
    bgApp.Stop(Seconds(simTime));
    Ptr<Application> bgPtr = bgApp.Get(0);
    bgPtr->TraceConnectWithoutContext("Tx", MakeCallback([=](Ptr<const Packet> p) {
        auto packet = const_cast<Packet*>(p);
        PriorityTag tag;
        tag.SetPriority(0);
        packet->AddPacketTag(tag);
      }));
  }

  // --- Install FlowMonitor ---
  FlowMonitorHelper flowmon;
  Ptr<FlowMonitor> monitor = flowmon.InstallAll();

  // --- Run simulation ---
  Simulator::Stop(Seconds(simTime));
  Simulator::Run();

  // --- Collect statistics ---
  monitor->CheckForLostPackets();
  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
  std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

  std::ofstream outFile("latency_results.csv");
  outFile << "FlowId,SourceAddress,SourcePort,DestAddress,DestPort,TxPackets,RxPackets,MeanDelay(ms),StdDevDelay(ms),MaxDelay(ms),LostPackets\n";

  for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); i != stats.end(); ++i)
  {
    FlowId id = i->first;
    FlowMonitor::FlowStats s = i->second;
    Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(id);

    double meanMs = (s.rxPackets > 0) ? s.delaySum.GetSeconds() / s.rxPackets * 1000.0 : 0.0;
    double stddevMs = 0.0;
    if (s.rxPackets > 1)
    {
      double meanSec = s.delaySum.GetSeconds() / s.rxPackets;
      stddevMs = std::sqrt((s.delaySumSquared.GetSeconds() / s.rxPackets) - (meanSec * meanSec)) * 1000.0;
    }
    double maxMs = s.maxDelay.GetSeconds() * 1000.0;

    outFile << id << ","
            << t.sourceAddress << "," << t.sourcePort << ","
            << t.destinationAddress << "," << t.destinationPort << ","
            << s.txPackets << ","
            << s.rxPackets << ","
            << meanMs << ","
            << stddevMs << ","
            << maxMs << ","
            << s.lostPackets << "\n";
  }
  outFile.close();

  // Optional: Export XML trace for per-packet analysis
  // monitor->SerializeToXmlFile("delays.xml", true, true);

  Simulator::Destroy();
  return 0;
}
