# It Helps To Network

In this module, we will demonstrate how to build secure networks in the cloud.

## Learning Outcomes

* Explore basic networking in the cloud.
* Discuss how to build virtual private clouds (VPCs).
* Explain the use of public and private IP addresses in the cloud.
* Describe the Google Network.
* Explain the role of firewall rules and routes.
* Explore hybrid cloud networking options including virtual private networks (VPNs), interconnect, and direct peering.
* Differentiate between load balancing options in the cloud.


Link to this section can be found [here](https://youtu.be/vQSOTKRVhK8).

---

# Introduction to Networking in the Cloud

Computers communicate with each other through a network. A computer in a single network like an office are connected to Local Area Network (LAN). Multiple locations can have their LANs connected to a Wide Area Network (WAN). 

Most networks today are connected to the Internet. This enables personal computers, servers, smartphone and other devices to communicate, provide and consume IT services.

![google_networking](https://media.discordapp.net/attachments/984655726406402088/985382726993584148/unknown.png?width=1246&height=701)

GCP uses state-of-the-art Software Defined Networking (SDN) and distributed systems technologies to host and deliver services around the world. When every millisecond of latency counts, Google ensures that content is delivered with the highest throughput.

Link to this section can be found [here](https://youtu.be/FvjyJu25C8I).

# Defining a Virtual Private Cloud

Virtual Private Cloud networks (VPCs) are used to build private networks on top of the larger Google network. With VPCs, users can apply many of the same security and access control rules as if they are building a physical network.

![vpc](https://media.discordapp.net/attachments/984655726406402088/985383860005117982/unknown.png?width=1246&height=701)

Subnets (sub-networks) are regional resources. They must be created in VPC networks to define sets of usable IP ranges for instances. VMs in different zones within the same region can share the same subnet.

Subnets are defined by an internal IP address prefix range and are specified as CIDR (Classless Inter-Domain Route) notations. IP ranges cannot overlap between subnets. They can be expanded but can never shrink.

![subnet_and_ip](https://media.discordapp.net/attachments/984655726406402088/985388145879187456/unknown.png?width=1246&height=701)

From the diagram above, `subnet1` is defined as `10.240.0.0/24` in the `us-west1` region. 2 VM instances in `us-west1-a` zone are in this subnet. Their IP addresses both come from the availale range of addresses in `subnet1`. 

Notice that one of the VM instances in `subnet3` (IP address `10.2.0.2`) belongs to `us-east1-a` zone, wihle the other VM instance in the same subnet (IP adress `10.2.0.3`) belongs to `us-east-2` zone.

Since subnets are regional resources, the instances can have their network interfaces associated with any subnet in the same region that contains their zones. While IP ranges are specific to one subnet, they can cross zones within the region. Users can also create multiple subnets in a single region.

Although subnets don't need to conform to a hierarchical IP scheme, the internal IP ranges for a subnet must conform to [RFC 1918](https://netbeez.net/blog/rfc1918/). 

A single VPN can be used to give private connectivity from a physical data center to the VPC.

![network_behavior](https://media.discordapp.net/attachments/984655726406402088/985389495690080276/unknown.png?width=1248&height=701)

VMs that are in different region but in the same VPC can communicate privately. From the diagram above, `VM1` and `VM2` can communicate at a local level even though they are separated geographically.

VMs that reside in different VPCs even if the subnets are in the same region, need to communicate via the internet. From the diagram above `VM3` and `VM4` will need public IP addresses to traverse the internet.

Networks don't communicate with any other network by default.

GCP offers two type of VPC networks determined by their subnet creation.

![auto_vs_custom_network](https://media.discordapp.net/attachments/984655726406402088/985391445571698688/unknown.png?width=1246&height=701)

In addition to the automatically created subnets, users can add more subnets manually to Auto Subnet Mode in chosen regions using IP ranges outside the set of predefined IP ranges. 

Conversion can be made from Auto Subnet Mode to Custom Subnet Mode. However, this conversion is one-way because Custom Mode networks cannot be changed to Auto Mode networks.

Link to this section can be found [here](https://youtu.be/b7RgndnUVi8).

# Public and Private IP Address Basics

A VPC is made up of subnets. Subnets need to be configured with a private IP CIDR address range. The private IP addresses are only used for internal network communication, and cannot be routed to the Internet.

![subnets_and_ip](https://media.discordapp.net/attachments/984655726406402088/985393417175592960/unknown.png?width=1246&height=701)

The number at the end of the range in an IPv4 address determines how many IP addresses are available with a CIDR address as it only freezes certain amount of binary digits in the 4 octets.

![ip_range](https://media.discordapp.net/attachments/984655726406402088/985393864384843836/unknown.png?width=1246&height=701)

Each time the number of static bits increase by 1, the number of available IP addresses decrease by half.

![public_private_ip](https://media.discordapp.net/attachments/984655726406402088/985394338609659964/unknown.png?width=1246&height=701)

VMs are unaware of their public IP addresses, they will only show their private IP addresses.

Link to this section can be found [here](https://youtu.be/LPm8DuCNug8).

# Google's Network Architecture

Below are some primary Google networking products:

1. Virtual Private Cloud (VPC)
    * VPC is a comprehensive set of networking capabilities and infrastructures managed by Google. 
    * With VPC, users can connect to GCP resources in a VPC and isolate them from each other for purposes of security, compliance and development vs test vs production environments.

2. Cloud Load Balancer
    * Cloud Load Balancer provides high-performance, scalable load balancing for GCP to ensure consistent performance for users.

3. Cloud Content Delivery Network (Cloud CDN)
    * Cloud CDN serves content to users with high availability and high performance. It is usually used to store files closer to the user. 
    * With it, Google's global network provides low latency and low cost content delivery.

4. Cloud Interconnect
    * Cloud Interconnect allows users to connect their own infrastructure to Google's Network edge with enterprise grade connections. 
    * The connections are offered by Google's partner network service providers, and may offer higher service levels than standard internet connections.

5. Cloud Domain Name System (Cloud DNS)
    * Cloud DNS translates requests for domain names into IP addresses. 
    * Google provides the infrastructure to publish specific domain names in high-volume DNS service suitable for production applications.

![google_network](https://media.discordapp.net/attachments/984655726406402088/985398291107823646/unknown.png?width=1246&height=701)

A region is a specific geographical location where users can run their resources. From the diagram above, the number in each region represents the number of zones within that region.

The Points of Presence (PoPs) are where the Google network is connected to the rest of the internet. By operating an extensive global network of interconnection points, GCP can bring its traffic close to its peer, thereby reducing costs and providing users with better experiences.

The blue lines are Google's global private network. This network connects regions and PoPs and is composed of hundreds of thousands of miles of fiber-optic cables and several submarine cable investments.

Link to this section can be found [here](https://youtu.be/vQ2P_9xRNRM).

# Routes and Firewall Rules in the Cloud

![route_maps](https://media.discordapp.net/attachments/984655726406402088/985398643177701437/unknown.png?width=1248&height=701)

However, manually created networks do not have such firewall rules, so users must create the rules manually.

![route_maps](https://media.discordapp.net/attachments/984655726406402088/985399537155854396/unknown.png?width=1246&height=701)

Routes match packets by destination IP addresses. However, no traffic will flow without also matching a firewall rule. 

A route is created when a network is created, enabling traffic delivery from anywhere. A route is also created when a subnet is created, this allows VMs in the same network to communicate.

Each route in the routes' collection can apply to one or more instanecs. A route applies to an instance if the network and instance tags match. If the network matches but there are no instance tags specified, the route applies to all instances in that network.

Compute Engine will then use the routes' collection to create individual read-onlyrouting tables for each instance.

![instance_routing](https://media.discordapp.net/attachments/984655726406402088/985400907153952808/unknown.png?width=1246&height=701)

The diagram above shows a massively scalable virtual router at the core of each network. Every VM instance in the network is directly connected to that router, and all packets leaving the VM instance are first handled at this layer before forwarding to the next hop.

The virtual network router selects the next hop for a packet by consulting the routing table for that instance.

![network_route](https://media.discordapp.net/attachments/984655726406402088/985401515353190410/unknown.png?width=1246&height=701)

Every route consists of a destination and a next hop. Traffic whose destination IP is within the destination range is sent to the next hop for delivery.

GCP firewall rules protect VM instances from unapproved connections both inbound and outbound known as ingress and egress respectively.

![gcp_firewall](https://media.discordapp.net/attachments/984655726406402088/985402059400556585/unknown.png?width=1248&height=701)

Users should express their desired firewall confguration as a set of firewall rules. Conceptually, a firewall rule is composed of certain parameters:

* Direction of the Rule
    * Inbound connections are matched against ingress rules only.
    * Outbound connections are matched against egress rules only.
* Source or Destination of the Connection
    * For ingress direction, sources can be specified as part of the rule with IP addresses, source tags or a source service account.
    * For egress direction, destinations can be specified as oart of the rule with one or more ranges of IP addresses.
* Protocol and Port of the Connection
    * Any rule can be restricted to apply to specified protocols or specific combinations of protocols and ports only.
* Action of the Rule
    * Allow or deny packets that match the direction, protocol, port and source or destination of the rule.
* Priority of the Rule
    * Governs the order in which the rules are evaluated.
* Rule Assignment
    * By default, all rules are assigned to all instances, but users can assign certain certain rules to certain instances only. 

## GCP Firewall Use Cases

Egress firewall rules control outgoing connections that originated inside of user's GCP network. 

Destinations to which a rule applies may be specified using IP CIDR ranges. Specifically, users can use destination range to protect from undesired connections initiated by a VM instance towards an external destinations. 

![egress_usecase](https://media.discordapp.net/attachments/984655726406402088/985403704511463535/unknown.png?width=1248&height=701)

Ingress firewall rules protect against incoming connections to the instance from any source.

Users can control ingress connections from VM instances by constructing inbound connection conditions using the conditions shown in the diagram below.

![ingress_usecase](https://media.discordapp.net/attachments/984655726406402088/985404516256071710/unknown.png?width=1246&height=701)

Link to this section can be found [here](https://youtu.be/b6P6mJnfMbo).

# Multiple VPC Networks

Shared VPC network allows an organization to connect resources from multiple projects to a common VPC network. This allows resources to communicate with each other securely and efficiently using internal IP addresses from that network.

![shared_vpc_network](https://media.discordapp.net/attachments/984655726406402088/985406697721970698/unknown.png?width=1246&height=701)

The diagram above shows a single network belonging to the Web Application Server's project. This network is shared with three other projects (Recommendation Service, Personalization Service, Analytics Service). 

Each of the service projects has instances that are in the same network as the Web Application Server. Private communication can be established with that server using internal IP addresses.

The Web Application Server communicates with clients and on-premises devices using the server's external IP address. The backend services on the other hand cannot be reached externally as they only communicate using internal IP addresses.

When user uses Shared VPC, they designate a host project and attach one or more other service projects to it. In this case, the Web Application Server's project is the host project, and the three other projects are service projects.

## VPC Network Peering

VPC network peering is a decentralized or distributed approach to multi-project networking, because each VPC network may remain under the control of separate administrative group, and retain the global firewall and routing tables associated.

VPC network peering does not incure the network latency, security, and cost drawbacks of using external IP addresses or VPNs.

VPC Network Peering allows private RFC 1918 connectivity across two VPC networks regardless whether they belong to the same project or organization. Remember that each VPC network will have firewall rules that define what traffic is allow, or denied between networks.

![network_peering](https://media.discordapp.net/attachments/984655726406402088/985408017270992906/unknown.png?width=1246&height=701)

The diagram above illustrates two organizations that represent a consumer and a producer. Each organization has its own organization node, VPC network, VM instance, network admin and instance admin.

To properly establish a VPC network peering, the producer network admin must peer the producer network with the consumer network and vice versa. 

After both peering connections are created, the VPC Network Peering session becomes active and routes are exchanged. This allow the VM instances to communicate privately using internal IP addresses.

![network_peering_considerations](https://media.discordapp.net/attachments/984655726406402088/985409163507146772/unknown.png?width=1246&height=701)

One point to remember is that only directly peered networks can communicate. This means that transitive peering is not supported. 

For example, if VPC network `N1` is peered to `N2` and `N3`, but `N2` and `N3` are not directly connected, VPC network `N2` cannot communicate with `N3` through peering. This is important if `N1` is a SaaS organization offering services to `N2` and `N3`.

![sharedvpc_vpcpeering](https://media.discordapp.net/attachments/984655726406402088/985410313375588393/unknown.png?width=1246&height=701)

Link to this section can be found [here](https://youtu.be/BPfbmg-PNbE).

# Lab: VPC Networking Fundamentals

In this lab, we will:

* Create custom mode VPC networks with firewall rules.
* Create VM instances using Compute Engine.
* Explore the connectivity for VM instances across VPC networks.
* Create a VM instance with multiple network interfaces.

Link to this section can be found [here](https://youtu.be/WDhftB0qqXQ).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1164919/labs/107700).

# Lab: Controlling Access on VPC Networks

In this lab, we will:

* Create an NGINX web server.
* Create tagged firewall rules.
* Create a service account with IAM roles.
* Explore permissions for the Network Admin and Security Admin roles.

Link to this section can be found [here](https://youtu.be/nGCSlph5mDQ).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1164919/labs/107703).

# Building Hybrid Clouds Using VPNs, Interconnecting, and Direct Peering

Cloud VPN securely connects on-premises network to a GCP VPC network through an IPsec VPN connection. Traffic travelling between the two networks is encrypted by one VPN gate, and then decrpyted by another VPN gate. This protects data as it travels over the public internet.

![cloud_vpn](https://media.discordapp.net/attachments/984655726406402088/985426498943070238/unknown.png?width=1246&height=701)

Cloud VPN is not a suitable use case if client's computers need to connect to a VPN using client VPN software.

## Interconnect Options

Cloud Interconnect provides two options for extending on-premises network to a GCP VPC network:

* Cloud Interconnect - Dedicated (aka Dedicated Interconnect)
* Cloud Interconnect - Partner (aka Partner Interconnect)

Both options allows access between resources by internal IP addresses in an on-premises network and VPC network. The choice will depends on connection requirements like location and connection capacity. 

![dedicated_interconnect](https://media.discordapp.net/attachments/984655726406402088/985427782412697610/unknown.png?width=1246&height=701)

Dedicated Interconnect provides direct physical conenctivity between an organization's on-premises network and the Google Cloud network edge. This allows large amounts of data to be transferred between networks. 

This can be more cost effective than purchasing additional bandwidth over the public internet.

![partner_interconnect](https://media.discordapp.net/attachments/984655726406402088/985428522778624010/unknown.png?width=1246&height=701)

If 10 or 100 Gbps connections aren't required, Partner Interconnect offers various capacity options. Additionally, if an organization cannot be physically meet Google's network requirements in a colocation facility, Partner Interconnect can allow connection to various service providers to access the VPC networks.

Partner Interconnect provides connectivity between on-premises network and the cloud network edge through a service provider, allowing an organization to extend the private network into its cloud network.

The service provider can provide solutions that minimizes router requirements on the organization's premises to only support an Ethernet interface to the cloud.

![comparisons_interconnect](https://media.discordapp.net/attachments/984655726406402088/985429841631391814/unknown.png?width=1246&height=701)

It is recommended to start with VPN tunnels and switch to Dedicated Interconnect or Partner Interconnect depending on the proximity to the colocationfacility and capacity requirements.

## Peering Options

There are two options for peering.

![direct_peering](https://media.discordapp.net/attachments/984655726406402088/985430385334845480/unknown.png?width=1246&height=701)

If an organization needs access to Google's public infrastructure, but doesn't meet the peering requirements, the organization can connect through a Carrier Peering service provider.

![carrier_peering](https://media.discordapp.net/attachments/984655726406402088/985430895131500574/unknown.png?width=1246&height=701)

Below are the comparison between the two peering options listed above.

![comparison_peering](https://media.discordapp.net/attachments/984655726406402088/985431324934410250/unknown.png?width=1246&height=701)

Link to this section can be found [here](https://youtu.be/va4799g7bGg).

# Different Options for Load Balancing

![load_balance_options](https://media.discordapp.net/attachments/984655726406402088/985431614433660968/unknown.png?width=1246&height=701)

Link to this section can be found [here](https://youtu.be/vV1nJzAB-lg).

# Lab: HTTP Load Balancer with Cloud Armor

In this lab, we will:

* Create HTTP and health check firewall rules.
* Configure two instance templates.
* Create two managed instance groups.
* Configure an HTTP load balancer with IPv4 and IPv6.
* Stress test an HTTP load balancer.
* Blacklist an IP address to restrict access to an HTTP load balancer.

Link to this section can be found [here](https://youtu.be/HfLnHi9Bu9A).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1164919/labs/107708).

# Lab: Create an Internal Load Balancer

In this lab, we will:

* Create HTTP and health check firewall rules.
* Configure two instance templates.
* Create two managed instance groups.
* COnfigure and test an internal load balancer.

Link to this section can be found [here](https://youtu.be/D2E81Ext_ig).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1164919/labs/107711).

---

# Module Quiz

1. What is a key distinguishing feature of networking in Google Cloud?

* [ ] Unlike other cloud providers, access lists and firewall rules are available.
* [X] **Network topology is not dependent on the IP address layout.**
* [ ] Network topology is dependent on IP address layout.
* [ ] IPV4 is supported.

> Feedback: Correct, network topology isn’t dependent on the IP address layout.

2. Select the global load balancer from the list.

* [ ] Internal
* [ ] Network
* [ ] Elastic
* [X] **TCP Proxy**

> Feedback: The global load balancer is a TCP proxy.

3. Which one of the following is true?

* [X] **VPCs are global and subnets are regional.**
* [ ] VPCs are regional and subnets are zonal.
* [ ] VPCs are regional. Subnets are not used in Google Cloud.
* [ ] Both VPCs and subnets are global.

> Feedback: VPCs are global and subnets are regional.

---

# Module Summary

Link to this section can be found [here](https://youtu.be/Lj1JQzOGG-0).