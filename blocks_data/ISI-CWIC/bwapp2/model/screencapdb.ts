/**========================================================
 * Module: screencapdb.ts
 * Created by wjwong on 10/21/15.
 =========================================================*/
/// <reference path="../server/typings/meteor/meteor.d.ts" />

interface iScreenCaps {
  _id: string,
  created: number,
  public: boolean,
  data: string
}

declare var ScreenCaps:any;
ScreenCaps = new Mongo.Collection('screencaps');
