/**========================================================
 * Module: genstatesdb.js
 * Created by wjwong on 9/11/15.
 =========================================================*/
/// <reference path="../server/typings/meteor/meteor.d.ts" />

declare var cBlockDecor:any;

cBlockDecor = class ciBlockDecor{
  static digit = 'digit';
  static logo = 'logo';
  static blank = 'blank';
};

/*interface iBlockDecor{
  digit: string,
  logo: string,
  blank: string
}*/

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

declare var GenStates:Mongo.Collection<iGenStates>;
GenStates = new Mongo.Collection<iGenStates>('genstates');

