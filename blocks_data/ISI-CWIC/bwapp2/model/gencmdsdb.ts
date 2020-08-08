/**========================================================
 * Module: gencmdsdb
 * Created by wjwong on 1/26/16.
 =========================================================*/
/// <reference path="../server/typings/meteor/meteor.d.ts" />
/// <reference path="genstatesdb.ts" />

interface iGenCmds {
  _id: string,
  block_meta: iBlockMeta,
  block_state: iBlockState[],
  utterance: string[],
  public: boolean,
  created: number,
  creator: string,
  name: string
}


declare var GenCmds:any;
GenCmds = new Mongo.Collection('gencmds');
