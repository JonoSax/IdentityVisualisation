This data models the movement of people within an organisation (ie people joining, moving departments/managerial levels and leaving).

Data specicifics:
- Created three new permissions for modelling purposes:
	- Management access: ManagerAccess0, ManagerAccess1, ManagerAccess2. The manager attribute of the identity will also change
	- New joiner accesses: NewJoiner0, NewJoiner1, NewJoiner2. They will have missing ID information until they have all joiner entitlements
- New identities should gain the new joiner access (first 0, then 1 then 2 in seperate time intervals) then get their full permissions only, but not always
- As people are promoted they get management access incrementally added sometimes or all at once
- When people leave, all their access is revoked immediately
- When people change departments they should only get the access of that department but this may not always be the case

Identity	RBAC test	Description of action
User0000	Mover (up)	Promoted to manager, will gain the manager access all at once 
User0001	Mover (up)	Promoted to manager, will gain the manager access all at once
User0002	Mover (up)	Promoted to manager, will gain the manager access incrementally
User0006	Mover (up)	Promoted to manager, will gain the manager access incrementally (in a different order)
User0007	Joiner 		Will be a new joiner, starting as a normal employee
User0008	Joiner (priv)	Will be a new joiner, starting as a manager (access all at once)
User0009	Joiner		Will be a new joiner starting in their allocated department (access gradually)
User0016	Mover		Will change to department 0 and immediately lose all their access and gain the that of the department (the 50 most common accesses of dept 0)
User0017	Mover		Will gradually gain some of the dept0 specific permissions and lose their dept1 permissions
User0021	Mover (ovProv)	Will change to department 0 but instead of losing their access, will just acquire the new access on top of their existing access
User0018	Mover		Will change to department 1. For this person this essentially means they just lose access to some systems as dept1 is less permissive than their current state
User0028	Leaver		Is a leaver and will lose all their access immediately but will remain on the identity exports
User0029	Leaver		Is a leaver and will lose all their access gradually but will remain on the identity exports
User0030	Leaver		Leaver and will lose all their access immediately but will also be removed from the identity exports
