# You Can't Secure The Cloud, Right?

In this module, we will outline how security in the cloud is administered in GCP.

## Learning Outcomes

* Describe the shared security model of the cloud
* Discuss Google's security responsibility versus a customer's responsibility
* Explore the different encryption options with GCP
* Identify best practices when configuring authentication and authorization using Cloud IAM

Link to this section can be found [here](https://youtu.be/aqqE06hOMtY).

---

# Introduction to Security in the Cloud

Google believes that security empowers innovation: if security is set first, everything else will follow.

![google_security](https://media.discordapp.net/attachments/984655726406402088/985103053562724423/unknown.png?width=1246&height=701)

Designing for security is pervasive throughout the entire Google's infrastructure, and security is paramount. Google invested heavily in its technical infrastructure and has dedicated engineers tasked with providing a secure and robust platform.

Countless organizations have lost data due to a security incident. A single breach can cost millions in fines and lost business, but a more serious data breach can permanently damage an organization's reputation with the loss of customer's trust.

However, many organizations don't have access to the resources they need to implement state-of-the-art security controls and techniques. GCP allows organizations to leverage Google's technical infrastructure to help secure their services and data for the entire information processing lifecycle including:

* Deployment of Services
* Storage of Data
* Communication between Services
* Operations by Administrators

Security cannot be an afterthought. It must be fundamental in all designs. Google builds security in progressive layers that deliver true defense-in-depth, it does not rely on one single technology to make it secure.

![security_layers](https://media.discordapp.net/attachments/984655726406402088/985104778176638996/unknown.png?width=1246&height=701)

## Hardware Infrastructure

Google build and design their own data centers that incorporate multiple layers of physical security protections. Access to the data centers are limited to a very small fraction of Google's employees. 

Google design their own server networking equipment and hardware security chips in those data centers. The servers use cryptographic signatures to make sure they're booting the correct software at the correct version, in the correct data center.

![hardware_layer](https://media.discordapp.net/attachments/984655726406402088/985105699078041651/unknown.png?width=1246&height=701)

## Service Deployment

Google Service Deployment provides the fabric for Google Cloud. When services communicate with one another, they do so via a remote procedure call (RPC). Google's infrastructure provides cryptographic privacy and integrity for all RPC calls for service-to-service communication.

Google also has an external Bug Bounty Program where third-party security researchers and developers can gain monetary rewards for finding vulnerabilities in Google's software components.

![service_layer](https://media.discordapp.net/attachments/984655726406402088/985106658034323486/unknown.png?width=1246&height=701)

Link to this section can be found [here](https://youtu.be/6JU4R0fstyg).

# The Shared Security Model

When users build and deploy an application their on-premises infrastructure, they are responsible for the security from the physical security of hardware and the premises which they are housed, through the data encryption on disk, the integrity of the network, all the way up to securing the content stored in those applications.

When a user moves an application to Google Cloud, Google will handle many of those lower-level layers of security like physical security, disk encryption and networking integrity. The upper layer of the security stack including data security remains as user's responsibility.

![security_responsibility](https://media.discordapp.net/attachments/984655726406402088/985107990665367622/unknown.png?width=1246&height=701)

Data access is almost always user's responsibility. Users will control who or what has access to their data at any time. Google Cloud provides tools like Cloud IAM to help users control these access, but they must be configured properly to protect the data.

Link to this section can be found [here](https://youtu.be/iFZY3qP0bSo).

# Encryption Options

There are several encryption options available on Google Cloud. They ranged from simple but limited control to complex but greater control and flexibility.

![options](https://media.discordapp.net/attachments/984655726406402088/985109809869242398/unknown.png?width=1246&height=701)

There is a fourth option, which is to encrypt the data locally before storing it in the cloud. This is often called as client-side encryption.

## Default Encryption (Only On GCP)

By default, GCP will encrypt data in transit and at rest:

* Data in transit is encrypted via TLS.
* Data at rest is encrypted with AES-256 key.

The encryption happens automatically.

## Customer-Managed Encryption Keys (CMEK)

Users manage their own encryption keys that protect data on Google Cloud. Google Cloud's Key Management Service (Cloud KMS) automates and simplifies the generation and management of encryption keys. 

The keys are managed by customers, but they never leave Google Cloud. 

Cloud KMS supports encryption, decryption, signing and verifying data. It supports both symmetric and asymmetric cryptographic keys, and a variety of popular algorithms.

Cloud KMS allows users to both rotate keys manually and to automate key rotations on a time-based interval.

## Customer-Supplied Encryption Keys (CSEK)

Users have more control over their keys but with greater management complexity. Users will use their own AES-256 encryption keys with GCP services. Users are responsible for generating and storing the keys.

Users have to provide the keys as part of their GCP API calls.

Google Cloud will use the provided key to encrypt the data before persisting it. The key will only exist in memory, and is discarded immediately after using.

### Persistent Disk Encryption with CSEK

Persistent disks such as those backing VMs can be encrypted with CSEK. Without CSEK and CMEK, those persistent disks are still encrypted with Google's default encryption. 

![persistent_disk_encryption](https://media.discordapp.net/attachments/984655726406402088/985112703192432640/unknown.png?width=1246&height=701)

### Other Persistent Disk Encryption Options

Users can also create their own persistent disk and redundantly encrypt them.

## Client-Side Encryption

With Client-side Encryption, data is encrypted before sending to Google Cloud. So, neither of the unencrypted data nor the decryption keys ever leave user's local device.

Link to this section can be found [here](https://youtu.be/DIY0R1OGiIQ).

# Authentication and Authorization with Cloud IAM

Cloud IAM enables cloud administrators to authorize "who can do what on which resources".

![partof_iam_policies](https://media.discordapp.net/attachments/984655726406402088/985114262097784852/unknown.png?width=1246&height=701)

Many users get started by logging into the GCP Console with a personal Gmail account. To collaborate with their teammates, they use Google Groups to gather people who are in the same role together. 

This approach is easy to get started but the team's identities aren't centrally manage. For example, if someone leaves the team, there is no central way to remove their access to the cloud resources immediately.

* GCP users that are also G Suite users can define Google Cloud policies in terms of G Suite users and groups. 
    * When someone leaves the team, the administrator can immediately disable their account using the Google Cloud Admin Console for G Suite. 

* GCP users who are not G Suite users can gain these same capabilities through Cloud Identity. 
    * It allows users and groups to be managed using the Google Cloud Admin Console, but the G Suite collaboration products like Gmail, Docs, Drive and Calendar aren't included. 
    * For this reason, Cloud Identity is available for free.

## Cloud Identity

Cloud Identity is a unified identity access and device management platform. 

![cloud_identity](https://media.discordapp.net/attachments/984655726406402088/985117998572531732/unknown.png?width=1246&height=701)

If users already have a centralized user management and identity system like Microsoft Active Directory or LDAP, Google Cloud's directory sync can help. This tool synchronizes users and groups from an existing Active Directory, mapping them in a Cloud Identity domain. 

However, the synchronization is only one way. This is because Cloud Directory Sync cannot modify informations ion Microsoft Active Directory or LDAP systems. The sync is scheduled to run without supervision on a fixed interval.

![cloud_directory_sync](https://media.discordapp.net/attachments/984655726406402088/985116506666987580/unknown.png?width=1246&height=701)

## IAM Roles

It is a collection of IAM permissions. Permissions are very low-level and fine grained. For example, to manage a VM, the user need to have permissions to create, delete, stop, start and change an instance.

To make the process easier, permissions are often grouped together into an IAM role for easier management. There are built in roles available for all GCP users, but users can also build and customise their own roles for their organization.

![iam_roles](https://media.discordapp.net/attachments/984655726406402088/985119318054424596/unknown.png?width=1246&height=701)

### IAM Primitive Roles

It apply across all GCP resources in a project. These primitive roles include owner, editor, viewer and billing admin.

![primitive_roles](https://media.discordapp.net/attachments/984655726406402088/985119912982888478/unknown.png?width=1246&height=701)

Billing Admins only have access to the billing information, but they don't have access to the resources.

### IAM Predefined Roles

It offers more fine-grained permissions on particular services. It only applies to a particular GCP service in a project. GCP services offer their own set of predefined roles and they define where those roles can be applied. 

### IAM Custom Roles

For some organizations, the primitive and predefined IAM rules may not offer enough granularity. IAM Custom Roles allow users to create their own custom roles composed of very granular permissions.

For now, custom roles can only be applied at project and organization level. It is currently not possible to apply them in the folder level.

## Access to Specific Items in the Hierarchy

When a user give service account permissions on a specific element of the resource hierarchy, the resulting policy applies to the element chosen, as well as elements below that resource in the hierarchy.

## Service Accounts

To allow interaction between services, they need to have an identity. Service accounts are used to authenticate service-to-service communication. Users can give a role level access from one service to another.

Suppose a user has an application running in a VM that needs to access data in Cloud Storage and only wants that VM to have access to hat data. The user can create an authorized service account to access that data, and attach the service account to that VM.

Service accounts are named with an email address, often ends with `gserviceaccount.com`.

Service accounts are also resources. So, it can have IAM policies attached to it. Therefore, different roles can view, edit or manage the service accounts.

Service account permissions can be changed without recreating the VM.

## Managing VM Permissions

Users can grant VM different identities. This makes it easier to manage different project permissions across applications.

Link to this section can be found [here](https://youtu.be/wv7STym4sd0).

# Lab: User Authentication on Cloud Identity-Aware Proxy (IAP)

Cloud IAP allows users to establish a central authorizational layer for applications over TLS. So, users can use application-level control models instead of relying on network level firewalls.

Applications and resources protected by IAP can only be accessed through the proxy by users and groups with the appropriate Cloud IAM role. The proxy provides a layer of protection between outside world and the internal service.

When a user is authorized to access an application or resource by Cloud IAP, they are subject to the fine-grained access controls implemented by the product in-use without the need for a VPN. Cloud IAP performs authentication and authorization checks when a user attempts to access an IAP-secured resource.

However, Cloud IAP does not protect against activities inside the VMs like accessing by a user via SSH. Cloud IAP also does not protect against activities within a project like VM-to-VM communication over the local network.

In this lab, we will:

* Write and deploy a simple App Engine app using `Python`.
* Enable and disable Cloud IAP to restrict acecess to the app.
* Get user identity information from Cloud IAP into the app.
* Cryptographically verify information from CLoud IAP to protect against spoofing.

Link to this section can be found [here](https://youtu.be/xhiqCANjww0).
<br>Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1162044/labs/107602).

# Identify Best Practices for Authorization using Cloud IAM

![iam_bestprac](https://media.discordapp.net/attachments/984655726406402088/985129634691711006/unknown.png?width=1246&height=701)

When it comes to service accounts, here are a few best practices:

![service_bestprac](https://media.discordapp.net/attachments/984655726406402088/985129949717467166/unknown.png?width=1246&height=701)

Link to this section can be found [here](https://youtu.be/47kGNm2Sb9Y).

# Lab: Cloud IAM Qwik Start

Link to the lab at [here](https://www.cloudskillsboost.google/course_sessions/1162044/labs/107605).

---

# Module Quiz

1. How are user identities created in Cloud IAM?

* [ ] User Identities are inherited from Active Directory by default
* [ ] User identities are created in the Cloud Identity Console in Google Cloud
* [ ] User identities are created in the Cloud IAM section of the console
* [X] **User identities are created outside of Google Cloud using a Google-administered domain**

> Feedback: Creating users and groups within Google Cloud is not possible.

2. Which of the following is not an encryption option for Google Cloud?

* [ ] Customer-supplied encryption Keys (CSEK)
* [ ] Customer-managed encryption keys (CMEK)
* [X] **Scripted encryption keys (SEK)**
* [ ] Google encryption by default

> Feedback: Scripted encryption keys is not an option with Google Cloud.

3. If a Cloud IAM policy gives you Owner permissions at the project level, your access to a resource in the project may be restricted by a more restrictive policy on that resource.

* [ ] True
* [X] **False**

> Feedback: Policies are a union of the parent and the resource. If a parent policy is less restrictive, it overrides a more restrictive resource policy.

---

# Module Summary

Link to this section can be found [here](https://youtu.be/FeRyYdS9MhM).