# ISI-CWIC

This document goes over the installation of the CWC project for the Block World application.

#### System Recommendations
We recommend the use of Ubuntu 14.04 with 2CPU, 2GB of Memory and 40GB storage.  We currently use Digitalocean.com as our hosting service.

#### Installation
Follow the steps to get block world working on your linux system.
[MongoDB](https://www.digitalocean.com/community/tutorials/how-to-install-mongodb-on-ubuntu-14-04)

#### Nodejs
Follow the directions for [“How To Install Using NVM”](https://www.digitalocean.com/community/tutorials/how-to-install-node-js-on-an-ubuntu-14-04-server)

We use version v5.10.0

1. Meteor `curl https://install.meteor.com | /bin/sh`
2. Git `apt-get install git`
3. Application, Clone Git Repo: `git clone https://github.com/danielmarcu/ISI-CWIC.git`
4. Install Node Packages:
  * cd ~/ISI-CWIC/bwapp2/server
  * npm install
  * cd ~/ISI-CWIC/bwapp2/client/vendor
  * npm install
5. Edit User List.  The application requires user authentication to create jobs and worlds.  Edit the file in: `~/ISI-CWIC/bwapp2/server/loadUsers.js` And put into the userlist the set of users you would like to enable with password.
6. App Run
 * `cd ~/ISI-CWIC/bwapp2`
 * `./start.sh`
7. Use web browser to navigate to:
 * `http://ipaddr:4000`
8. App Stop
 * `cd ~/ISI-CWIC/bwapp2`
 * `./kill -9 [m]eteor`

### Usage
Block World has many options.  The main focus is in creating worlds for annotation and simulation.  To that end we will cover the usage of Block World for states>generate 
<img src="BlockWorldHowtoImages/generate.png" width=100/>

The following image shows the Block Generation UI:


<img src="BlockWorldHowtoImages/Create.png" width=400/>

The user interface consists of a 3 main components. The Scene which renders the block world, the Creation/Review area, and setting up the block type to render.

Import States
Import states option takes a json (described below) file describing the states of a block world and renders them and gives the user the ability to save the block world into the database for later view.  There are sample files (17_Num1_phys.json, 5blocksavefinal.json, 20blocklayout.json) located in ‘ISI-CWIC/bwapp2/Sample’.

Create Custom Layout
The user can create their own block layouts by clicking on the “Create Custom Layout”.  This option allows the user to arrange the 20 blocks as needed for the project.  Click on the button <img src="BlockWorldHowtoImages/custom.png" width=100/> to get started.  This is the workspace for the simulation to create the image from the blocks:


<img src="BlockWorldHowtoImages/20blocks.png" width=400/>


Scene
The scene shows the blocks available to use about 20 of them with different logos on them.   This is where you manipulate the blocks and the camera.  The camera is what you are using to view the scene.  In the next section we will describe the navigation.  

<img src="BlockWorldHowtoImages/20blocks_world.png" width=300/>

Navigation
There are a number of keys and buttons to push to help you move the camera around and move the blocks around.

<img src="BlockWorldHowtoImages/Navigation.png" width=200/>


The arrow keys will move the camera forward, backward, left and right.  To pivot the camera so you can see up or down just hold down the mouse button in the scene AWAY FROM BLOCKS and move the mouse around.  The camera will move while the mouse button is held.
To move a block click on a block and you will see a red column highlight the block and now you can drag it around the table.  Holding down the shift key while hold down the mouse button will lock the block in place and allow you to move the block up or down.  Useful for stack blocks.  We will ignore the alt keys for rotation as there is a simpler way to rotate.

Special Buttons
This section contains special buttons for managing the scene.

<img src="BlockWorldHowtoImages/SpecialButtons.png" width=200/>

Reset Camera - allows you to reset the camera to its original place when you moved it and don’t want it there anymore

Capture Frame - captures the current scene into a frame so that you can move on to the next step in creating the mage. - we will talk about how to use this in the Getting Started section.

Reset Frame - allows you to reset to the current worked on image to the last capture point so if you screw up it will only be for one step. 

On Physics - is a button to turn physics on or off.  The simulation has physics, it sometimes doesn’t work very well with some stacking layout as it will make all the blocks fall.  You can turn off Physics if it makes it easier to create the layout in the simulation.

Rotate Cube - this is where you will rotate a cube if the simulation requires it.  In our sample image in the introduction one cube requires rotation by 45 degrees in the Y axis.  Now a little bit about axis.  For a cube that is in their default position - the Y axis is up and down, the Z axis is forward and backward, and the X axis it left to right.  Take some time to play around with these rotations to see what they do so you know how to use them when needed in the simulation.
Here are some sample for  45 degree rotation:
Y: <img src="BlockWorldHowtoImages/Y.png" width=100/> X: <img src="BlockWorldHowtoImages/X.png" width=100/> Z: <img src="BlockWorldHowtoImages/Z.png" width=100/>

Delete #Cubes - when starting, count the number of cubes that you will need to use.  Then use this option to delete the rest of them.  It doesn’t matter which cubes you delete as long as there are only the same number of cubes as in the image.  If you move cubes around capture a frame then delete cubes you will lose data as it resets all of the cubes! 

JSON Layout
The major benefit of the simulation system is the ability for a user to import a json block layout file and run physics on the system.  To this point sample json file describes a 20 block (max blocks) system with 1 scene.  The file is located in ISI-CWIC/bwapp2/Sample/20blocklayout.json.  The main features for the JSON description are described in the typescript file ISI-CWIC/bwapp2/model/genstatesdb.ts - the JSON description starts with the iGenStates interface.

```js
cBlockDecor = class ciBlockDecor{
 static digit = 'digit';
 static logo = 'logo';
 static blank = 'blank';
};

interface iGenStates {
 _id: string,
 block_meta: iBlockMeta,
 block_states: iBlockStates[],
 type?: string,
 public: boolean,
 created: number,
 creator: string,
 name: string
}

interface iBlockStates{
 created?: number,
 screencapid?: string,
 enablephysics?: boolean,
 block_state: iBlockState[]
}

interface iBlockState{
 id: number,
 position: iPosRot,
 rotation?: iPosRot
}

interface iPosRot{
 [x: string]: number
}

interface iBlockMeta {
 decoration?: string,
 savefinalstate?: boolean,
 blocks: Array<iBlockMetaEle>
}

interface iBlockMetaEle
{
 name: string,
 id: number,
 shape: iShapeMeta
}

interface iShapeParams{
 face_1: iFaceEle,
 face_2: iFaceEle,
 face_3: iFaceEle,
 face_4: iFaceEle,
 face_5: iFaceEle,
 face_6: iFaceEle,
 side_length: number
}
interface iShapeMeta{
 type: string,
 size: number,
 shape_params: iShapeParams
}

interface iFaceEle{
 color: string,
 orientation: number
}
```
