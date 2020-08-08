/**
 * Created by wjwong on 12/16/15.
 */
/// <reference path="../server/typings/meteor/meteor.d.ts" />
/// <reference path="genstatesdb.ts" />

interface iGenExps {
  _id: string,
  block_meta: iBlockMeta,
  block_state: iBlockState[],
  utterance: string[],
  public: boolean,
  created: number,
  creator: string,
  name: string
}


declare var GenExps:any;
GenExps = new Mongo.Collection('genexps');
