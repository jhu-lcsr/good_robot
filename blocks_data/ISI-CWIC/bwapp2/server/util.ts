/**========================================================
 * Module: util
 * Created by wjwong on 1/26/16.
 =========================================================*/
/// <reference path="./typings/meteor/meteor.d.ts" />

declare var isRole;
isRole = function(usr, role){
  if(usr){
    if(role == 'guest'){ //there is always a default guest account via artwells:accounts-guest
      if(usr && usr.profile && usr.profile.guest) return usr.profile.guest;
    } else if(usr && usr.profile && usr.profile.roles)
      return (usr.profile.roles.indexOf(role) > -1);
  }
  return false;
};

declare var rndInt;
rndInt = function (min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};
